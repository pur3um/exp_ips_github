"""Lazy Q Update optimizer with optional projected-matrix EMA.

Variant 3 of the progressive-rank family:
- keeps the auto-init logic and cosine rank growth from auto_cos_inc_rank.py
- updates the randomized low-rank basis Q only every K steps (or when rank changes)
- keeps Newton-Schulz / projected polar refinement every step
- optionally applies EMA to the projected matrix B = Q^T X before Newton-Schulz
- keeps all other defaults / update rules unchanged when EMA is disabled

progressive cosine rank는 그대로 유지
lazy Q refresh는 그대로 유지
B = Q^T X에 대한 EMA를 선택적으로 추가
기본값은 꺼져 있어서 기존 동작 유지

Place this file under optims/auto_lazy_q_update.py.
"""

import math
import statistics
from typing import Optional

import torch


def zeropower_via_newtonschulz5(
    G: torch.Tensor,
    steps: int,
    eps: float = 1e-7,
    use_bfloat16: bool = True,
) -> torch.Tensor:
    assert G.ndim == 2

    a, b, c = (3.4445, -4.7750, 2.0315)

    compute_dtype = torch.bfloat16 if (use_bfloat16 and G.is_cuda) else torch.float32
    X = G.to(dtype=compute_dtype)

    transposed = False
    if X.size(-2) > X.size(-1):
        X = X.mT
        transposed = True

    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)

    for _ in range(int(steps)):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if transposed:
        X = X.mT

    return X.to(dtype=G.dtype)


@torch.no_grad()
def zeropower_via_lowrank_matrix_sign(
    G: torch.Tensor,
    steps: int = 5,
    rank: int = 200,
    oversample: int = 4,
    eps: float = 1e-6,
    small_ns_bfloat16: bool = False,
    rescale: bool = False,
) -> torch.Tensor:
    assert G.ndim == 2, "Expected a 2D matrix after any conv flattening."

    if rank <= 0:
        return torch.zeros_like(G)

    X = G.float()
    transposed = False

    if X.size(-2) > X.size(-1):
        X = X.mT
        transposed = True

    m, n = X.shape
    r = min(m, n)
    sketch_dim = min(rank + max(0, oversample), r)

    if sketch_dim >= r:
        Z = zeropower_via_newtonschulz5(
            X,
            steps=steps,
            eps=eps,
            use_bfloat16=small_ns_bfloat16,
        ).float()
        if transposed:
            Z = Z.mT
        return Z.type_as(G)

    Omega = torch.randn(n, sketch_dim, device=X.device, dtype=X.dtype)
    Y = X @ Omega

    Q, _ = torch.linalg.qr(Y, mode="reduced")
    B = Q.mT @ X

    S = zeropower_via_newtonschulz5(
        B,
        steps=steps,
        eps=eps,
        use_bfloat16=small_ns_bfloat16,
    ).float()

    Z = Q @ S

    if rescale and sketch_dim > 0:
        Z = Z * math.sqrt(float(r) / float(sketch_dim))

    if transposed:
        Z = Z.mT

    return Z.type_as(G)



def _round_up_to_multiple(value: int, multiple: int = 8) -> int:
    multiple = max(1, int(multiple))
    return int(math.ceil(int(value) / float(multiple)) * multiple)



def _clamp_rank(value: int, floor_rank: int, ceil_rank: int) -> int:
    floor_rank = int(floor_rank)
    ceil_rank = max(floor_rank, int(ceil_rank))
    return max(floor_rank, min(int(value), ceil_rank))


@torch.no_grad()
def build_muon_search_matrix(
    grad: torch.Tensor,
    momentum: torch.Tensor,
    beta: float = 0.95,
    nesterov: bool = True,
) -> torch.Tensor:
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum

    if update.ndim == 4:
        update = update.view(len(update), -1)
    elif update.ndim > 2:
        update = update.view(update.shape[0], -1)
    elif update.ndim < 2:
        update = update.view(1, -1)

    return update


@torch.no_grad()
def preview_muon_search_matrix(
    grad: torch.Tensor,
    momentum: torch.Tensor,
    beta: float = 0.95,
    nesterov: bool = True,
) -> torch.Tensor:
    """Preview the Muon search matrix without mutating the live momentum buffer."""
    momentum_preview = momentum.detach().clone()
    grad_preview = grad.detach().clone()

    momentum_preview.lerp_(grad_preview, 1 - beta)
    update = grad_preview.lerp_(momentum_preview, beta) if nesterov else momentum_preview

    if update.ndim == 4:
        update = update.view(len(update), -1)
    elif update.ndim > 2:
        update = update.view(update.shape[0], -1)
    elif update.ndim < 2:
        update = update.view(1, -1)

    return update


@torch.no_grad()
def choose_auto_rank_start(
    update: torch.Tensor,
    floor_rank: int,
    probe_rank: int,
    energy_tau: float = 0.90,
    round_multiple: int = 8,
    eps: float = 1e-12,
) -> int:
    assert update.ndim == 2

    X = update.float()
    if X.size(-2) > X.size(-1):
        X = X.mT

    m, n = X.shape
    limit = min(m, n)
    floor_rank = max(1, min(int(floor_rank), limit))
    probe_rank = max(floor_rank, min(int(probe_rank), limit))

    Omega = torch.randn(n, probe_rank, device=X.device, dtype=X.dtype)
    Y = X @ Omega

    Q, _ = torch.linalg.qr(Y, mode="reduced")
    B = Q.mT @ X

    svals = torch.linalg.svdvals(B.float())
    if svals.numel() == 0:
        return floor_rank

    capture = torch.cumsum(svals.square(), dim=0) / ((X.square().sum()) + eps)

    if float(capture[-1].item()) < float(energy_tau):
        return probe_rank

    threshold = torch.tensor(float(energy_tau), device=capture.device, dtype=capture.dtype)
    r_hat = int(torch.searchsorted(capture, threshold).item()) + 1
    r_hat = max(floor_rank, r_hat)
    r_hat = min(r_hat, probe_rank)
    r_hat = _round_up_to_multiple(r_hat, round_multiple)
    return _clamp_rank(r_hat, floor_rank, probe_rank)



def get_cosine_rank(step: int, start_rank: int, end_rank: int, warmup_steps: int) -> int:
    step = int(step)
    start_rank = int(start_rank)
    end_rank = int(end_rank)
    warmup_steps = int(warmup_steps)

    if warmup_steps <= 1:
        return end_rank
    if step <= 1:
        return start_rank
    if step >= warmup_steps:
        return end_rank

    t = (step - 1) / float(warmup_steps - 1)
    progress = 0.5 * (1.0 - math.cos(math.pi * t))

    rank = start_rank + (end_rank - start_rank) * progress
    return int(round(rank))


@torch.no_grad()
def zeropower_via_lazy_lowrank_matrix_sign(
    G: torch.Tensor,
    state: dict,
    *,
    current_step: int,
    refresh_gap: int,
    steps: int = 5,
    rank: int = 200,
    oversample: int = 4,
    eps: float = 1e-6,
    small_ns_bfloat16: bool = False,
    rescale: bool = False,
    use_b_ema: bool = False,
    b_ema_decay: float = 0.9,
) -> torch.Tensor:
    assert G.ndim == 2, "Expected a 2D matrix after any conv flattening."

    if rank <= 0:
        return torch.zeros_like(G)

    X = G.float()
    transposed = False
    if X.size(-2) > X.size(-1):
        X = X.mT
        transposed = True

    m, n = X.shape
    r = min(m, n)
    sketch_dim = min(rank + max(0, oversample), r)

    if sketch_dim >= r:
        state.pop("lazy_q_B_ema", None)
        state["lazy_q_b_ema_initialized"] = False
        Z = zeropower_via_newtonschulz5(
            X,
            steps=steps,
            eps=eps,
            use_bfloat16=small_ns_bfloat16,
        ).float()
        if transposed:
            Z = Z.mT
        state["lazy_q_last_refresh_step"] = int(current_step)
        state["lazy_q_refresh_this_step"] = True
        return Z.type_as(G)

    refresh_gap = max(1, int(refresh_gap))
    use_b_ema = bool(use_b_ema)
    b_ema_decay = float(b_ema_decay)
    if use_b_ema and not (0.0 <= b_ema_decay < 1.0):
        raise ValueError(f"lazy_q_b_ema_decay must be in [0, 1), got {b_ema_decay}")

    cached_Q = state.get("lazy_q_cache", None)
    cached_shape = state.get("lazy_q_shape", None)
    cached_sketch_dim = state.get("lazy_q_sketch_dim", None)
    cached_transposed = state.get("lazy_q_transposed", None)
    cached_rank = state.get("lazy_q_rank", None)
    last_refresh_step = int(state.get("lazy_q_last_refresh_step", 0))

    need_refresh = (
        cached_Q is None
        or cached_shape != (m, n)
        or cached_sketch_dim != sketch_dim
        or cached_transposed != transposed
        or cached_rank != int(rank)
        or (int(current_step) - last_refresh_step) >= refresh_gap
    )

    if need_refresh:
        Omega = torch.randn(n, sketch_dim, device=X.device, dtype=X.dtype)
        Y = X @ Omega
        Q, _ = torch.linalg.qr(Y, mode="reduced")
        state["lazy_q_cache"] = Q.detach()
        state["lazy_q_shape"] = (m, n)
        state["lazy_q_sketch_dim"] = sketch_dim
        state["lazy_q_transposed"] = transposed
        state["lazy_q_rank"] = int(rank)
        state["lazy_q_last_refresh_step"] = int(current_step)
        state["lazy_q_refresh_this_step"] = True
        if use_b_ema:
            state.pop("lazy_q_B_ema", None)
            state["lazy_q_b_ema_initialized"] = False
    else:
        state["lazy_q_refresh_this_step"] = False

    Q = state["lazy_q_cache"]
    B = Q.mT @ X

    if use_b_ema:
        ema_buf = state.get("lazy_q_B_ema", None)
        if (
            bool(state.get("lazy_q_refresh_this_step", False))
            or ema_buf is None
            or tuple(ema_buf.shape) != tuple(B.shape)
            or ema_buf.device != B.device
            or ema_buf.dtype != B.dtype
        ):
            ema_buf = B.detach().clone()
        else:
            ema_buf.mul_(b_ema_decay).add_(B, alpha=(1.0 - b_ema_decay))
        state["lazy_q_B_ema"] = ema_buf
        state["lazy_q_b_ema_initialized"] = True
        B_for_sign = ema_buf
    else:
        state.pop("lazy_q_B_ema", None)
        state["lazy_q_b_ema_initialized"] = False
        B_for_sign = B

    S = zeropower_via_newtonschulz5(
        B_for_sign,
        steps=steps,
        eps=eps,
        use_bfloat16=small_ns_bfloat16,
    ).float()
    Z = Q @ S

    if rescale and sketch_dim > 0:
        Z = Z * math.sqrt(float(r) / float(sketch_dim))

    if transposed:
        Z = Z.mT

    return Z.type_as(G)


@torch.no_grad()
def muon_update(
    grad: torch.Tensor,
    momentum: torch.Tensor,
    state: dict,
    beta: float = 0.95,
    ns_steps: int = 5,
    nesterov: bool = True,
    rank: int = 200,
    oversample: int = 4,
    lowrank_rescale: bool = False,
    eps: float = 1e-6,
    small_ns_bfloat16: bool = False,
    step: int = 1,
    lazy_q_update_gap: int = 100,
    lazy_q_use_b_ema: bool = False,
    lazy_q_b_ema_decay: float = 0.9,
    current_rank: Optional[int] = None,
) -> torch.Tensor:
    update = build_muon_search_matrix(
        grad,
        momentum,
        beta=beta,
        nesterov=nesterov,
    )

    applied_rank = int(rank if current_rank is None else current_rank)
    update = zeropower_via_lazy_lowrank_matrix_sign(
        update,
        state,
        current_step=step,
        refresh_gap=lazy_q_update_gap,
        steps=ns_steps,
        rank=applied_rank,
        oversample=oversample,
        eps=eps,
        small_ns_bfloat16=small_ns_bfloat16,
        rescale=lowrank_rescale,
        use_b_ema=lazy_q_use_b_ema,
        b_ema_decay=lazy_q_b_ema_decay,
    )

    update *= max(1.0, update.size(-2) / update.size(-1)) ** 0.5
    return update



def adam_update(grad, buf1, buf2, step, betas, eps):
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0] ** step)
    buf2c = buf2 / (1 - betas[1] ** step)
    return buf1c / (buf2c.sqrt() + eps)


class SingleDeviceAutoLazyQWithAuxAdam(torch.optim.Optimizer):
    """Muon + auxiliary Adam with lazy Q refresh every K steps and optional B-EMA."""

    def __init__(self, param_groups):
        normalized_groups = []
        for group in param_groups:
            assert "use_muon" in group, "Each param group must include use_muon=True/False."

            g = dict(group)
            if g["use_muon"]:
                g.setdefault("lr", 0.003)
                g.setdefault("momentum", 0.95)
                g.setdefault("weight_decay", 0.0)
                g.setdefault("ns_steps", 5)
                g.setdefault("nesterov", True)
                g.setdefault("rank", 200)
                g.setdefault("rank_start", g["rank"])
                g.setdefault("rank_end", g["rank"])
                g.setdefault("warmup_steps", 1)
                g.setdefault("lazy_q_update_gap", 100)
                g.setdefault("lazy_q_use_b_ema", False)
                g.setdefault("lazy_q_b_ema_decay", 0.9)
                g.setdefault("oversample", 4)
                g.setdefault("lowrank_rescale", False)
                g.setdefault("eps", 1e-6)
                g.setdefault("small_ns_bfloat16", False)
                g.setdefault("step", 0)
                g.setdefault("current_rank", g["rank_start"])
                g.setdefault("current_target_rank", max(int(g["rank"]), int(g["rank_end"])))
                g.setdefault("current_method", "cosine_lazy_q_closed_form")
                g.setdefault("current_lazy_q_gap", g["lazy_q_update_gap"])
                g.setdefault("current_lazy_q_last_refresh_step", 0)
                g.setdefault("current_lazy_q_refresh_this_step", False)
                g.setdefault("current_lazy_q_use_b_ema", bool(g["lazy_q_use_b_ema"]))
                g.setdefault("current_lazy_q_b_ema_decay", float(g["lazy_q_b_ema_decay"]))
                g.setdefault("current_lazy_q_b_ema_initialized", False)

                g.setdefault("auto_init_rank_start", False)
                g.setdefault("init_probe_steps", 8)
                g.setdefault("init_energy", 0.90)
                g.setdefault("init_round_multiple", 8)
                g.setdefault("auto_rank_start_final", None)
                g.setdefault("_init_rank_candidates", [])
            else:
                g.setdefault("lr", 3e-4)
                g.setdefault("betas", (0.9, 0.95))
                g.setdefault("eps", 1e-10)
                g.setdefault("weight_decay", 0.0)

            normalized_groups.append(g)

        super().__init__(normalized_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                group_step = int(group.get("step", 0)) + 1
                group["step"] = group_step

                if bool(group.get("auto_init_rank_start", False)) and group.get("auto_rank_start_final") is None:
                    if group_step <= int(group["init_probe_steps"]):
                        step_candidates = []
                        for p in group["params"]:
                            if p.grad is None:
                                continue
                            state = self.state[p]
                            if len(state) == 0:
                                state["momentum_buffer"] = torch.zeros_like(p)

                            search_matrix = preview_muon_search_matrix(
                                p.grad,
                                state["momentum_buffer"],
                                beta=group["momentum"],
                                nesterov=group["nesterov"],
                            )
                            candidate = choose_auto_rank_start(
                                search_matrix,
                                floor_rank=group["rank"],
                                probe_rank=group["rank_end"],
                                energy_tau=group["init_energy"],
                                round_multiple=group["init_round_multiple"],
                            )
                            step_candidates.append(int(candidate))

                        if step_candidates:
                            step_median = int(statistics.median(step_candidates))
                            step_median = max(int(group["rank"]), step_median)
                            step_median = min(int(group["rank_end"]), step_median)
                            group["_init_rank_candidates"].append(step_median)

                            provisional_start = int(statistics.median(group["_init_rank_candidates"]))
                            provisional_start = max(int(group["rank"]), provisional_start)
                            provisional_start = min(int(group["rank_end"]), provisional_start)
                            group["rank_start"] = _clamp_rank(
                                _round_up_to_multiple(provisional_start, int(group["init_round_multiple"])),
                                int(group["rank"]),
                                max(int(group["rank"]), int(group["rank_end"])),
                            )

                    if group_step == int(group["init_probe_steps"]):
                        if group["_init_rank_candidates"]:
                            final_start = int(statistics.median(group["_init_rank_candidates"]))
                            final_start = max(int(group["rank"]), final_start)
                            final_start = min(int(group["rank_end"]), final_start)
                            final_start = _clamp_rank(
                                _round_up_to_multiple(final_start, int(group["init_round_multiple"])),
                                int(group["rank"]),
                                max(int(group["rank"]), int(group["rank_end"])),
                            )
                        else:
                            final_start = int(group["rank"])
                        group["auto_rank_start_final"] = final_start
                        group["rank_start"] = final_start

                scheduled_rank = get_cosine_rank(
                    step=group_step,
                    start_rank=group["rank_start"],
                    end_rank=group["rank_end"],
                    warmup_steps=group["warmup_steps"],
                )
                applied_rank = max(int(group["rank"]), int(scheduled_rank))

                group["current_rank"] = applied_rank
                group["current_target_rank"] = max(int(group["rank"]), int(group["rank_end"]))
                group["current_method"] = (
                    "cosine_lazy_q_auto_start"
                    if bool(group.get("auto_init_rank_start", False))
                    else "cosine_lazy_q_closed_form"
                )
                if bool(group.get("lazy_q_use_b_ema", False)):
                    group["current_method"] += "_b_ema"
                group["current_lazy_q_gap"] = int(group["lazy_q_update_gap"])
                group["current_lazy_q_use_b_ema"] = bool(group.get("lazy_q_use_b_ema", False))
                group["current_lazy_q_b_ema_decay"] = float(group.get("lazy_q_b_ema_decay", 0.9))

                refresh_happened = False
                last_refresh_step = 0
                b_ema_initialized = False
                for p in group["params"]:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)

                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)

                    update = muon_update(
                        p.grad,
                        state["momentum_buffer"],
                        state,
                        beta=group["momentum"],
                        ns_steps=group["ns_steps"],
                        nesterov=group["nesterov"],
                        rank=group["rank"],
                        oversample=group["oversample"],
                        lowrank_rescale=group["lowrank_rescale"],
                        eps=group["eps"],
                        small_ns_bfloat16=group["small_ns_bfloat16"],
                        step=group_step,
                        lazy_q_update_gap=group["lazy_q_update_gap"],
                        lazy_q_use_b_ema=group.get("lazy_q_use_b_ema", False),
                        lazy_q_b_ema_decay=group.get("lazy_q_b_ema_decay", 0.9),
                        current_rank=applied_rank,
                    )

                    refresh_happened = refresh_happened or bool(state.get("lazy_q_refresh_this_step", False))
                    last_refresh_step = max(last_refresh_step, int(state.get("lazy_q_last_refresh_step", 0)))
                    b_ema_initialized = b_ema_initialized or bool(state.get("lazy_q_b_ema_initialized", False))

                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])

                group["current_lazy_q_refresh_this_step"] = refresh_happened
                group["current_lazy_q_last_refresh_step"] = last_refresh_step
                group["current_lazy_q_b_ema_initialized"] = b_ema_initialized
            else:
                for p in group["params"]:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)

                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0

                    state["step"] += 1
                    update = adam_update(
                        p.grad,
                        state["exp_avg"],
                        state["exp_avg_sq"],
                        state["step"],
                        group["betas"],
                        group["eps"],
                    )

                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss
