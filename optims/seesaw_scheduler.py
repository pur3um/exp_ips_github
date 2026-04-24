import math
from bisect import bisect_right
from typing import Dict, List, Optional


class SeesawScheduler:
    """
    Seesaw scheduler adapted from Meterez et al. (ICLR 2026).

    Key design points implemented here:
      1) Boundary factor and actual LR factor are separated.
      2) Warmup is optional; for INR/NeRF we default to *no* warmup.
      3) Effective total serial iterations are recomputed using the *actual*
         capped batch size of each phase.
      4) We keep the original *ray budget* phase-by-phase as closely as possible.
         If phase rays are not divisible by the current batch size, we round up the
         serial step count and report the budget mismatch.

    Parameters
    ----------
    base_lr_adam, base_lr_muon:
        Peak learning rates for the Adam and Muon branches.
    base_N_rand:
        Baseline ray batch size.
    total_iters:
        Baseline training horizon in optimizer steps.
    boundary_factor:
        Factor gamma > 1 describing *where* the underlying cosine scheduler would
        cut the learning rate. If gamma=2, phase boundaries are the points where
        the cosine hits 1/2, 1/4, 1/8, ... of the peak LR.
    lr_decay_factor:
        Actual LR cut factor applied at each Seesaw boundary. If None, it is
        inferred from the NSGD/Adam equivalence line:
            lr_decay_factor * sqrt(batch_growth_factor) = boundary_factor.
        Example: boundary_factor=2, batch_growth_factor=2 -> lr_decay_factor=sqrt(2).
    batch_growth_factor:
        Per-phase batch growth factor beta.
    max_N_rand:
        Hard cap on the ray batch size.
    warmup_steps:
        Optional warmup steps before the Seesaw phases begin.
    cosine_variant:
        'paper'   -> eta(t) = eta0 * cos(pi t / (2T))
        'pytorch' -> eta(t) = eta0 * 0.5 * (1 + cos(pi t / T))
        The paper's continuous derivation uses the former. The latter matches the
        common PyTorch-style cosine annealing schedule.
    max_phases:
        Safety cap on phase count.
    min_lr_ratio:
        Lower stop threshold for boundary generation.
    """

    def __init__(
        self,
        base_lr_adam: float,
        base_lr_muon: float,
        base_N_rand: int,
        total_iters: int,
        boundary_factor: float = 2.0,
        lr_decay_factor: Optional[float] = None,
        batch_growth_factor: float = 2.0,
        max_N_rand: Optional[int] = None,
        warmup_steps: int = 0,
        cosine_variant: str = "paper",
        max_phases: int = 128,
        min_lr_ratio: float = 1e-6,
    ) -> None:
        if boundary_factor <= 1.0:
            raise ValueError(f"boundary_factor must be > 1.0, got {boundary_factor}")
        if batch_growth_factor <= 1.0:
            raise ValueError(f"batch_growth_factor must be > 1.0, got {batch_growth_factor}")
        if total_iters <= 0:
            raise ValueError(f"total_iters must be > 0, got {total_iters}")
        if base_N_rand <= 0:
            raise ValueError(f"base_N_rand must be > 0, got {base_N_rand}")
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")
        if warmup_steps >= total_iters:
            raise ValueError(
                f"warmup_steps ({warmup_steps}) must be smaller than total_iters ({total_iters})"
            )
        if cosine_variant not in {"paper", "pytorch"}:
            raise ValueError(
                f"cosine_variant must be one of ['paper', 'pytorch'], got {cosine_variant}"
            )

        self.base_lr_adam = float(base_lr_adam)
        self.base_lr_muon = float(base_lr_muon)
        self.base_N_rand = int(base_N_rand)
        self.total_iters = int(total_iters)
        self.boundary_factor = float(boundary_factor)
        self.batch_growth_factor = float(batch_growth_factor)
        self.lr_decay_factor = (
            float(lr_decay_factor)
            if lr_decay_factor is not None
            else float(self.boundary_factor / math.sqrt(self.batch_growth_factor))
        )
        if self.lr_decay_factor <= 1.0:
            raise ValueError(f"lr_decay_factor must be > 1.0, got {self.lr_decay_factor}")

        self.max_N_rand = (
            int(max_N_rand) if max_N_rand is not None else int(self.base_N_rand * 64)
        )
        self.warmup_steps = int(warmup_steps)
        self.cosine_variant = str(cosine_variant)
        self.max_phases = int(max_phases)
        self.min_lr_ratio = float(min_lr_ratio)

        self.original_total_rays = int(self.total_iters * self.base_N_rand)
        self.original_decay_iters = self.total_iters - self.warmup_steps

        self._ns_equivalence_product = self.lr_decay_factor * math.sqrt(self.batch_growth_factor)
        self._aggressive_limit = math.sqrt(self.batch_growth_factor)
        self._is_more_aggressive_than_paper_limit = self.lr_decay_factor + 1e-12 < self._aggressive_limit

        self._build_schedule()

    def _cosine_boundary_step(self, lr_ratio: float) -> float:
        ratio = float(lr_ratio)
        ratio = max(min(ratio, 1.0), 0.0)

        if self.cosine_variant == "paper":
            # eta(t) = eta0 * cos(pi t / (2T))
            return self.original_decay_iters * (2.0 * math.acos(ratio) / math.pi)

        # eta(t) = eta0 * 0.5 * (1 + cos(pi t / T))
        cos_arg = max(min(2.0 * ratio - 1.0, 1.0), -1.0)
        return self.original_decay_iters * (math.acos(cos_arg) / math.pi)

    def _generate_original_decay_boundaries(self) -> List[float]:
        boundaries = [0.0]
        prev_t = 0.0
        for k in range(1, self.max_phases + 1):
            target_ratio = self.boundary_factor ** (-k)
            if target_ratio <= self.min_lr_ratio:
                break
            t_k = self._cosine_boundary_step(target_ratio)
            if t_k <= prev_t + 1e-12:
                continue
            if t_k >= self.original_decay_iters - 1e-12:
                break
            boundaries.append(float(t_k))
            prev_t = float(t_k)
        boundaries.append(float(self.original_decay_iters))
        return boundaries

    def _build_schedule(self) -> None:
        original_decay_boundaries = self._generate_original_decay_boundaries()
        phase_original_steps: List[float] = []
        phase_serial_steps: List[int] = []
        phase_lr_adam: List[float] = []
        phase_lr_muon: List[float] = []
        phase_n_rand: List[int] = []
        phase_original_rays: List[int] = []
        phase_serial_rays: List[int] = []

        scheduled_boundaries = [float(self.warmup_steps)]

        # Warmup is kept at the original batch size and original serial horizon.
        if self.warmup_steps > 0:
            self.warmup_rays = int(self.warmup_steps * self.base_N_rand)
        else:
            self.warmup_rays = 0

        for k in range(len(original_decay_boundaries) - 1):
            duration_orig = float(original_decay_boundaries[k + 1] - original_decay_boundaries[k])
            rays_orig = int(round(duration_orig * self.base_N_rand))

            raw_batch = int(round(self.base_N_rand * (self.batch_growth_factor ** k)))
            batch_k = min(raw_batch, self.max_N_rand)
            serial_steps_k = max(1, int(math.ceil(float(rays_orig) / float(batch_k))))
            rays_serial_k = int(serial_steps_k * batch_k)

            phase_original_steps.append(duration_orig)
            phase_original_rays.append(rays_orig)
            phase_serial_steps.append(serial_steps_k)
            phase_serial_rays.append(rays_serial_k)
            phase_n_rand.append(batch_k)
            phase_lr_adam.append(self.base_lr_adam / (self.lr_decay_factor ** k))
            phase_lr_muon.append(self.base_lr_muon / (self.lr_decay_factor ** k))

            scheduled_boundaries.append(scheduled_boundaries[-1] + float(serial_steps_k))

        self.original_decay_boundaries = original_decay_boundaries
        self.phase_original_steps = phase_original_steps
        self.phase_original_rays = phase_original_rays
        self.phase_serial_steps = phase_serial_steps
        self.phase_serial_rays = phase_serial_rays
        self.phase_lr_adam = phase_lr_adam
        self.phase_lr_muon = phase_lr_muon
        self.phase_N_rand = phase_n_rand
        self.phase_seesaw_start = scheduled_boundaries[:-1]
        self.phase_seesaw_end = scheduled_boundaries[1:]
        self._total_seesaw_iters = int(round(scheduled_boundaries[-1]))

        self.total_serial_rays = int(self.warmup_rays + sum(self.phase_serial_rays))
        self.total_target_rays = int(self.original_total_rays)
        self.ray_budget_diff = int(self.total_serial_rays - self.total_target_rays)
        self.equal_ray_budget_exact = bool(self.ray_budget_diff == 0)

        warnings: List[str] = []
        if self._is_more_aggressive_than_paper_limit:
            warnings.append(
                "Configured lr_decay_factor is more aggressive than the paper's non-divergent limit "
                f"(lr_decay_factor={self.lr_decay_factor:.6g} < sqrt(beta)={self._aggressive_limit:.6g})."
            )
        if any(n >= self.max_N_rand for n in self.phase_N_rand):
            warnings.append(
                "N_rand cap was activated; equal-ray budget is recomputed with the capped batch size, "
                "so later speedup can be smaller than the uncapped Seesaw ideal."
            )
        if not self.equal_ray_budget_exact:
            warnings.append(
                "Equal-ray budget is not exact because phase ray counts are rounded to integer steps; "
                f"scheduled run processes {self.ray_budget_diff:+d} rays relative to the target budget."
            )
        self.budget_warning = " ".join(warnings)

    @property
    def num_phases(self) -> int:
        return len(self.phase_seesaw_start)

    @property
    def effective_total_iters(self) -> int:
        return self._total_seesaw_iters

    def phase_for_step(self, step: int) -> int:
        if self.num_phases == 0:
            return -1
        idx = bisect_right(self.phase_seesaw_start, float(step)) - 1
        if idx < 0:
            return 0
        if idx >= self.num_phases:
            return self.num_phases - 1
        return idx

    def step(self, global_step: int) -> Dict[str, float]:
        step = int(max(0, global_step))

        if self.warmup_steps > 0 and step < self.warmup_steps:
            scale = float(step + 1) / float(max(1, self.warmup_steps))
            return {
                "lr_adam": self.base_lr_adam * scale,
                "lr_muon": self.base_lr_muon * scale,
                "N_rand": self.base_N_rand,
                "phase": -1,
                "phase_name": "warmup",
            }

        if self.num_phases == 0:
            return {
                "lr_adam": self.base_lr_adam,
                "lr_muon": self.base_lr_muon,
                "N_rand": self.base_N_rand,
                "phase": 0,
                "phase_name": "constant",
            }

        k = self.phase_for_step(step)
        return {
            "lr_adam": self.phase_lr_adam[k],
            "lr_muon": self.phase_lr_muon[k],
            "N_rand": self.phase_N_rand[k],
            "phase": k,
            "phase_name": f"phase_{k}",
        }

    def describe(self) -> str:
        lines: List[str] = []
        lines.append(
            "[Seesaw] "
            f"boundary_factor={self.boundary_factor:.6g} "
            f"lr_decay_factor={self.lr_decay_factor:.6g} "
            f"batch_growth_factor={self.batch_growth_factor:.6g}"
        )
        lines.append(
            "[Seesaw] "
            f"nsgd_product=lr_decay_factor*sqrt(beta)={self._ns_equivalence_product:.6g} "
            f"(target boundary_factor={self.boundary_factor:.6g})"
        )
        lines.append(
            "[Seesaw] "
            f"cosine_variant={self.cosine_variant} warmup_steps={self.warmup_steps} "
            f"base_N_rand={self.base_N_rand} max_N_rand={self.max_N_rand} "
            f"original_total_iters={self.total_iters} effective_total_iters={self.effective_total_iters}"
        )
        lines.append(
            "[Seesaw] "
            f"target_rays={self.total_target_rays} scheduled_rays={self.total_serial_rays} "
            f"ray_budget_diff={self.ray_budget_diff:+d} exact={self.equal_ray_budget_exact}"
        )
        if self.budget_warning:
            lines.append(f"[Seesaw][WARN] {self.budget_warning}")
        lines.append(
            "[Seesaw] phase | serial_iters | original_steps(eqv) | lr_adam | lr_muon | N_rand"
        )
        if self.warmup_steps > 0:
            lines.append(
                f"        {'warmup':>5} | {self.warmup_steps:>12d} | {self.warmup_steps:>18d} | "
                f"linear -> {self.base_lr_adam:.3e} | linear -> {self.base_lr_muon:.3e} | {self.base_N_rand:d}"
            )
        for k in range(self.num_phases):
            lines.append(
                f"        {k:>5d} | {self.phase_serial_steps[k]:>12d} | "
                f"{self.phase_original_steps[k]:>18.2f} | {self.phase_lr_adam[k]:.3e} | "
                f"{self.phase_lr_muon[k]:.3e} | {self.phase_N_rand[k]:d}"
            )
        return "\n".join(lines)
