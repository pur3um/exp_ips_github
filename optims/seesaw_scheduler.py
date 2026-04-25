import math
from bisect import bisect_right
from typing import Dict, Optional


class SeesawScheduler:
    """
    INR-oriented Seesaw scheduler.

    Differences from the original draft:
      - boundary_decay_factor (cosine boundary locations) is separated from
        lr_decay_factor (the actual LR cut).
      - batch caps are folded into effective duration computation so the
        continuous equal-ray budget stays aligned phase-by-phase.
      - original-step -> effective-step mapping is exposed for rescaling
        optimizer-internal schedules such as progressive rank growth.
    """

    def __init__(
        self,
        base_lr_adam: float,
        base_lr_muon: float,
        base_N_rand: int,
        total_iters: int,
        boundary_decay_factor: float = 2.0,
        lr_decay_factor: Optional[float] = None,
        beta: float = 2.0,
        warmup_iters: int = 0,
        warmup_start_factor: float = 0.0,
        max_N_rand: Optional[int] = None,
        max_phases: int = 64,
        min_cos: float = -0.999,
    ) -> None:
        if total_iters <= 0:
            raise ValueError(f"total_iters must be > 0, got {total_iters}")
        if base_N_rand <= 0:
            raise ValueError(f"base_N_rand must be > 0, got {base_N_rand}")
        if boundary_decay_factor <= 1.0:
            raise ValueError(
                f"boundary_decay_factor must be > 1.0, got {boundary_decay_factor}"
            )
        if beta < 1.0:
            raise ValueError(f"beta must be >= 1.0, got {beta}")
        if warmup_iters < 0:
            raise ValueError(f"warmup_iters must be >= 0, got {warmup_iters}")
        if warmup_iters >= total_iters:
            raise ValueError(
                f"warmup_iters ({warmup_iters}) must be smaller than total_iters ({total_iters})"
            )
        if not (0.0 <= warmup_start_factor <= 1.0):
            raise ValueError(
                f"warmup_start_factor must be in [0, 1], got {warmup_start_factor}"
            )

        self.base_lr_adam = float(base_lr_adam)
        self.base_lr_muon = float(base_lr_muon)
        self.base_N_rand = int(base_N_rand)
        self.total_iters = int(total_iters)
        self.boundary_decay_factor = float(boundary_decay_factor)
        self.beta = float(beta)
        self.warmup_iters = int(warmup_iters)
        self.warmup_start_factor = float(warmup_start_factor)
        self.max_N_rand = (
            int(max_N_rand) if max_N_rand is not None else int(self.base_N_rand * 64)
        )
        self.max_phases = int(max_phases)
        self.min_cos = float(min_cos)

        if self.max_N_rand <= 0:
            raise ValueError(f"max_N_rand must be > 0, got {self.max_N_rand}")

        if lr_decay_factor is None:
            lr_decay_factor = self.boundary_decay_factor / math.sqrt(self.beta)
        if lr_decay_factor <= 1.0:
            raise ValueError(f"lr_decay_factor must be > 1.0, got {lr_decay_factor}")
        if lr_decay_factor + 1e-12 < math.sqrt(self.beta):
            raise ValueError(
                "lr_decay_factor is more aggressive than the Seesaw stability limit: "
                f"lr_decay_factor={lr_decay_factor:.6f} < sqrt(beta)={math.sqrt(self.beta):.6f}"
            )
        self.lr_decay_factor = float(lr_decay_factor)

        self.equivalence_product = self.lr_decay_factor * math.sqrt(self.beta)
        self.on_equivalence_line = math.isclose(
            self.equivalence_product,
            self.boundary_decay_factor,
            rel_tol=1e-6,
            abs_tol=1e-12,
        )

        self.decay_total_iters = self.total_iters - self.warmup_iters

        original_rel_boundaries = [0.0]
        for k in range(1, self.max_phases + 1):
            target_ratio = self.boundary_decay_factor ** (-k)
            cos_val = 2.0 * target_ratio - 1.0
            if cos_val <= self.min_cos:
                break
            t_k = self.decay_total_iters * math.acos(cos_val) / math.pi
            if t_k >= self.decay_total_iters:
                break
            original_rel_boundaries.append(t_k)
        original_rel_boundaries.append(float(self.decay_total_iters))

        self.original_boundaries = [float(self.warmup_iters) + b for b in original_rel_boundaries]

        effective_rel_boundaries = [0.0]
        phase_lr_adam = []
        phase_lr_muon = []
        phase_n_rand = []
        phase_uncapped_n_rand = []
        phase_original_durations = []
        phase_effective_durations = []
        cap_hit = False

        for k in range(len(original_rel_boundaries) - 1):
            duration_orig = original_rel_boundaries[k + 1] - original_rel_boundaries[k]
            uncapped_n_rand = int(round(self.base_N_rand * (self.beta ** k)))
            n_rand_k = min(uncapped_n_rand, self.max_N_rand)
            if n_rand_k < uncapped_n_rand:
                cap_hit = True

            duration_eff = duration_orig * (float(self.base_N_rand) / float(n_rand_k))
            effective_rel_boundaries.append(effective_rel_boundaries[-1] + duration_eff)

            phase_lr_adam.append(self.base_lr_adam / (self.lr_decay_factor ** k))
            phase_lr_muon.append(self.base_lr_muon / (self.lr_decay_factor ** k))
            phase_n_rand.append(int(n_rand_k))
            phase_uncapped_n_rand.append(int(uncapped_n_rand))
            phase_original_durations.append(float(duration_orig))
            phase_effective_durations.append(float(duration_eff))

        self.phase_original_start = original_rel_boundaries[:-1]
        self.phase_original_end = original_rel_boundaries[1:]
        self.phase_effective_start = effective_rel_boundaries[:-1]
        self.phase_effective_end = effective_rel_boundaries[1:]
        self.phase_lr_adam = phase_lr_adam
        self.phase_lr_muon = phase_lr_muon
        self.phase_N_rand = phase_n_rand
        self.phase_uncapped_N_rand = phase_uncapped_n_rand
        self.phase_original_durations = phase_original_durations
        self.phase_effective_durations = phase_effective_durations
        self.cap_hit = cap_hit

        self._effective_total_iters_float = float(self.warmup_iters) + effective_rel_boundaries[-1]
        self._effective_total_iters = max(1, int(math.ceil(self._effective_total_iters_float)))

    @property
    def num_phases(self) -> int:
        return len(self.phase_effective_start)

    @property
    def effective_total_iters(self) -> int:
        return self._effective_total_iters

    @property
    def effective_total_iters_float(self) -> float:
        return self._effective_total_iters_float

    def _warmup_progress(self, step: float) -> float:
        if self.warmup_iters <= 0:
            return 1.0
        progress = (float(step) + 1.0) / float(self.warmup_iters)
        return min(1.0, max(self.warmup_start_factor, progress))

    def phase_for_step(self, step: int) -> int:
        step = float(step)
        if step < self.warmup_iters:
            return -1
        rel_step = step - float(self.warmup_iters)
        idx = bisect_right(self.phase_effective_start, rel_step) - 1
        if idx < 0:
            return 0
        if idx >= self.num_phases:
            return self.num_phases - 1
        return idx

    def step(self, global_step: int) -> Dict[str, float]:
        if global_step < self.warmup_iters:
            factor = self._warmup_progress(global_step)
            return {
                "lr_adam": self.base_lr_adam * factor,
                "lr_muon": self.base_lr_muon * factor,
                "N_rand": self.base_N_rand,
                "phase": -1,
                "is_warmup": True,
            }

        k = self.phase_for_step(global_step)
        return {
            "lr_adam": self.phase_lr_adam[k],
            "lr_muon": self.phase_lr_muon[k],
            "N_rand": self.phase_N_rand[k],
            "phase": k,
            "is_warmup": False,
        }

    def map_original_to_effective_step(self, original_step: float) -> float:
        original_step = float(original_step)
        if original_step <= 0.0:
            return 0.0
        if original_step <= self.warmup_iters:
            return original_step
        if original_step >= self.total_iters:
            return float(self.effective_total_iters)

        rel_step = original_step - float(self.warmup_iters)
        idx = bisect_right(self.phase_original_start, rel_step) - 1
        idx = max(0, min(idx, self.num_phases - 1))

        rel_inside = rel_step - self.phase_original_start[idx]
        compression = float(self.base_N_rand) / float(self.phase_N_rand[idx])
        eff_rel = self.phase_effective_start[idx] + rel_inside * compression
        return float(self.warmup_iters) + eff_rel

    def rescale_schedule_steps(self, original_steps: int) -> int:
        return max(1, int(round(self.map_original_to_effective_step(original_steps))))

    def describe(self) -> str:
        lines = []
        lines.append(
            "[Seesaw] boundary_decay={:.6g} lr_decay={:.6g} beta={:.6g} "
            "equiv_product=lr_decay*sqrt(beta)={:.6g}".format(
                self.boundary_decay_factor,
                self.lr_decay_factor,
                self.beta,
                self.equivalence_product,
            )
        )
        lines.append(
            "[Seesaw] equivalence line: {}".format(
                "satisfied" if self.on_equivalence_line else "not exact"
            )
        )
        lines.append(
            f"[Seesaw] warmup_iters={self.warmup_iters} base_N_rand={self.base_N_rand} "
            f"max_N_rand={self.max_N_rand} original_total_iters={self.total_iters}"
        )
        if self.cap_hit:
            lines.append(
                "[Seesaw] batch cap became active. Effective durations were recomputed "
                "with duration *= base_N_rand / capped_N_rand, so the equal-ray budget is "
                "preserved phase-by-phase in continuous step space."
            )
        lines.append(
            "[Seesaw] phase | orig iters | seesaw iters | lr_adam | lr_muon | N_rand(capped/uncapped)"
        )
        for k in range(self.num_phases):
            lines.append(
                f"        {k:>5d} | {self.phase_original_durations[k]:>10.2f} | "
                f"{self.phase_effective_durations[k]:>12.2f} | "
                f"{self.phase_lr_adam[k]:.3e} | {self.phase_lr_muon[k]:.3e} | "
                f"{self.phase_N_rand[k]:d}/{self.phase_uncapped_N_rand[k]:d}"
            )
        lines.append(
            f"[Seesaw] total seesaw iters: {self.effective_total_iters} "
            f"(continuous={self.effective_total_iters_float:.2f}, original={self.total_iters})"
        )
        return "\n".join(lines)
