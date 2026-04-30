#!/usr/bin/env python3
"""
BoTorch-based hyper-parameter tuner for a NeRF PyTorch runner such as
`run_nerf_ranksched.py`.

Target hyper-parameters by default:
  - lr            -> --lrate         (log scale)
  - lr_decay      -> --lrate_decay   (integer, in 1000 steps in original NeRF)
  - muon_lr       -> --muon_lrate    (log scale)
  - muon_momentum -> --muon_momentum (linear scale)

The script launches one training process per trial, reads the selected metric
from <basedir>/<expname>/readme.txt, and uses BoTorch to propose the next trials.

Example:
  python botorch_nerf_tune.py \
    --runner /path/to/run_nerf_ranksched.py \
    --config configs/lego.txt \
    --hpo-dir hpo_runs/lego_aux_auto \
    --optimizer aux-sign-auto-cos-inc \
    --train-scheduler rank_wsd \
    --n-iters 100000 \
    --num-init 8 \
    --num-bo 24 \
    --q-batch 2 \
    --cuda-devices 0,1 \
    -- --dataset_type blender --white_bkgd --half_res
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

try:
    from botorch.acquisition.logei import qLogExpectedImprovement
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import SingleTaskGP
    from botorch.models.transforms import Standardize
    from botorch.optim import optimize_acqf
    from botorch.sampling.normal import SobolQMCNormalSampler
    from gpytorch.mlls import ExactMarginalLogLikelihood
except Exception as exc:  # pragma: no cover - import-time environment check
    raise SystemExit(
        "BoTorch / GPyTorch import failed. Install with: pip install botorch gpytorch\n"
        f"Original error: {exc}"
    )


DEFAULT_SEARCH_SPACE: List[Dict[str, Any]] = [
    {
        "name": "lr",
        "cli_arg": "--lrate",
        "kind": "log_float",
        "low": 1e-5,
        "high": 2e-3,
    },
    {
        "name": "lr_decay",
        "cli_arg": "--lrate_decay",
        "kind": "int",
        "low": 50,
        "high": 500,
    },
    {
        "name": "muon_lr",
        "cli_arg": "--muon_lrate",
        "kind": "log_float",
        "low": 5e-4,
        "high": 1e-2,
    },
    {
        "name": "muon_momentum",
        "cli_arg": "--muon_momentum",
        "kind": "float",
        "low": 0.80,
        "high": 0.98,
    },
]

METRIC_PATTERNS = {
    "testset_mean_psnr": re.compile(r"^testset_mean_psnr:\s*([-+0-9.eE]+)\s*$", re.MULTILINE),
    "testset_mean_loss": re.compile(r"^testset_mean_loss:\s*([-+0-9.eE]+)\s*$", re.MULTILINE),
    "testset_mean_ssim": re.compile(r"^testset_mean_ssim:\s*([-+0-9.eE]+)\s*$", re.MULTILINE),
    "testset_mean_lpips": re.compile(r"^testset_mean_lpips:\s*([-+0-9.eE]+)\s*$", re.MULTILINE),
    "current_train_psnr": re.compile(r"^current_train_psnr:\s*([-+0-9.eE]+)\s*$", re.MULTILINE),
    "current_train_loss": re.compile(r"^current_train_loss:\s*([-+0-9.eE]+)\s*$", re.MULTILINE),
}

VAL_PSNR_PATTERN = re.compile(r"\[VAL\]\s+Iter:\s*\d+\s+PSNR:\s*([-+0-9.eE]+)")
TRAIN_PSNR_PATTERN = re.compile(r"\[TRAIN\].*?PSNR:\s*([-+0-9.eE]+)")


@dataclass(frozen=True)
class TrialSpec:
    trial_id: int
    x_unit: List[float]
    params: Dict[str, Any]
    phase: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="BoTorch hyper-parameter tuner for run_nerf_ranksched.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--runner", required=True, help="Path to run_nerf_ranksched.py")
    p.add_argument("--config", required=True, help="NeRF config file passed to --config")
    p.add_argument("--hpo-dir", required=True, help="Directory for HPO state, trial logs, and NeRF runs")
    p.add_argument("--python", default=sys.executable, help="Python executable for training subprocesses")
    p.add_argument("--workdir", default=None, help="Working directory for training subprocesses. Default: runner parent")

    p.add_argument("--optimizer", default="aux-sign-auto-cos-inc")
    p.add_argument("--train-scheduler", default="rank_wsd", choices=["rank_wsd", "warmup_cosine", "exp_decay"])
    p.add_argument("--n-iters", type=int, default=100000)
    p.add_argument("--i-testset", type=int, default=0, help="0 means use --n-iters so readme.txt is produced at the end")
    p.add_argument("--i-print", type=int, default=2000)
    p.add_argument("--i-weights", type=int, default=10**9, help="Large default avoids checkpoint IO during HPO")
    p.add_argument("--seed", type=int, default=0, help="Base seed; trial_id is added")
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--allow-reload", action="store_true", help="Do not add --no_reload to trial commands")

    p.add_argument("--metric", default="testset_mean_psnr", choices=list(METRIC_PATTERNS.keys()) + ["last_val_psnr", "last_train_psnr"])
    p.add_argument("--mode", default="maximize", choices=["maximize", "minimize"])
    p.add_argument("--failure-objective", type=float, default=-1.0e9, help="Objective value assigned to failed trials")

    p.add_argument("--space-json", default=None, help="Optional JSON file overriding the search space")
    p.add_argument("--num-init", type=int, default=8, help="Sobol initialization trials")
    p.add_argument("--num-bo", type=int, default=24, help="Number of Bayesian optimization iterations")
    p.add_argument("--q-batch", type=int, default=1, help="Candidates proposed per BO iteration")
    p.add_argument("--raw-samples", type=int, default=256)
    p.add_argument("--num-restarts", type=int, default=20)
    p.add_argument("--mc-samples", type=int, default=256)
    p.add_argument("--cuda-devices", default="0", help="Comma-separated GPU ids. Use empty string for CPU/no CUDA_VISIBLE_DEVICES override")
    p.add_argument("--timeout-min", type=float, default=0.0, help="Per trial timeout. 0 disables timeout")
    p.add_argument("--resume", action="store_true", help="Resume from hpo-dir/trials.jsonl")
    p.add_argument("--exp-prefix", default="bo")

    p.add_argument(
        "training_args",
        nargs=argparse.REMAINDER,
        help="Extra args passed to the NeRF runner. Put them after '--'. Example: -- --dataset_type blender --white_bkgd",
    )
    return p.parse_args()


def strip_remainder_marker(args: Sequence[str]) -> List[str]:
    args = list(args)
    if args and args[0] == "--":
        args = args[1:]
    return args


def load_search_space(path: Optional[str]) -> List[Dict[str, Any]]:
    space = DEFAULT_SEARCH_SPACE if path is None else json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(space, list) or not space:
        raise ValueError("Search space must be a non-empty list.")
    names = set()
    for spec in space:
        for key in ("name", "cli_arg", "kind", "low", "high"):
            if key not in spec:
                raise ValueError(f"Search-space item missing {key}: {spec}")
        if spec["name"] in names:
            raise ValueError(f"Duplicate parameter name: {spec['name']}")
        names.add(spec["name"])
        if spec["kind"] not in {"float", "log_float", "int"}:
            raise ValueError(f"Unsupported kind for {spec['name']}: {spec['kind']}")
        if float(spec["low"]) >= float(spec["high"]):
            raise ValueError(f"low must be < high for {spec['name']}")
        if spec["kind"] == "log_float" and (float(spec["low"]) <= 0 or float(spec["high"]) <= 0):
            raise ValueError(f"log_float bounds must be positive for {spec['name']}")
    return [dict(s) for s in space]


def unit_to_params(x_unit: Sequence[float], space: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if len(x_unit) != len(space):
        raise ValueError("Dimension mismatch between x_unit and search space")
    params: Dict[str, Any] = {}
    for xi, spec in zip(x_unit, space):
        x = min(1.0, max(0.0, float(xi)))
        low = float(spec["low"])
        high = float(spec["high"])
        kind = spec["kind"]
        if kind == "log_float":
            value = 10.0 ** (math.log10(low) + x * (math.log10(high) - math.log10(low)))
            params[spec["name"]] = float(value)
        elif kind == "float":
            params[spec["name"]] = float(low + x * (high - low))
        elif kind == "int":
            params[spec["name"]] = int(round(low + x * (high - low)))
        else:
            raise ValueError(f"Unsupported kind: {kind}")
    return params


def params_to_cli(params: Dict[str, Any], space: Sequence[Dict[str, Any]]) -> List[str]:
    cli: List[str] = []
    for spec in space:
        name = spec["name"]
        arg = spec.get("cli_arg")
        if not arg:
            continue
        value = params[name]
        if isinstance(value, float):
            value_str = f"{value:.10g}"
        else:
            value_str = str(value)
        cli.extend([arg, value_str])
    return cli


def objective_from_metric(metric_value: Optional[float], mode: str, failure_objective: float) -> float:
    if metric_value is None or not math.isfinite(metric_value):
        return float(failure_objective)
    return float(metric_value if mode == "maximize" else -metric_value)


def parse_metric_from_files(
    exp_dir: Path,
    stdout_path: Path,
    metric: str,
) -> Tuple[Optional[float], str]:
    readme_path = exp_dir / "readme.txt"
    if metric in METRIC_PATTERNS and readme_path.exists():
        text = readme_path.read_text(encoding="utf-8", errors="replace")
        match = METRIC_PATTERNS[metric].search(text)
        if match:
            return float(match.group(1)), str(readme_path)

    # Fallback metrics from stdout/stderr log.
    if stdout_path.exists():
        text = stdout_path.read_text(encoding="utf-8", errors="replace")
        if metric == "last_val_psnr":
            matches = VAL_PSNR_PATTERN.findall(text)
            if matches:
                return float(matches[-1]), str(stdout_path)
        if metric == "last_train_psnr":
            matches = TRAIN_PSNR_PATTERN.findall(text)
            if matches:
                return float(matches[-1]), str(stdout_path)

    return None, ""


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    records: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def write_csv_summary(path: Path, records: Sequence[Dict[str, Any]], space: Sequence[Dict[str, Any]]) -> None:
    fieldnames = [
        "trial_id",
        "phase",
        "status",
        "metric",
        "metric_value",
        "objective",
        "returncode",
        "duration_sec",
        "expname",
        "cuda_device",
    ] + [spec["name"] for spec in space] + ["metric_source", "stdout_path", "exp_dir", "cmd"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            row = {k: r.get(k, "") for k in fieldnames}
            params = r.get("params") or {}
            for spec in space:
                row[spec["name"]] = params.get(spec["name"], "")
            writer.writerow(row)


def make_expname(prefix: str, trial_id: int, params: Dict[str, Any]) -> str:
    # Keep names readable and shell/path safe.
    fragments = [f"{prefix}_{trial_id:04d}"]
    for key in ("lr", "lr_decay", "muon_lr", "muon_momentum"):
        if key in params:
            value = params[key]
            if isinstance(value, float):
                fragments.append(f"{key}-{value:.3g}".replace("+", "").replace(".", "p"))
            else:
                fragments.append(f"{key}-{value}")
    return "_".join(fragments).replace("/", "_")


def run_one_trial(
    trial: TrialSpec,
    args: argparse.Namespace,
    space: Sequence[Dict[str, Any]],
    cuda_device: Optional[str],
    run_basedir: Path,
    log_dir: Path,
    records_path: Path,
) -> Dict[str, Any]:
    runner = Path(args.runner).resolve()
    expname = make_expname(args.exp_prefix, trial.trial_id, trial.params)
    exp_dir = run_basedir / expname
    stdout_path = log_dir / f"{expname}.log"
    timeout = None if args.timeout_min <= 0 else args.timeout_min * 60.0

    cmd = [
        args.python,
        str(runner),
        "--config",
        args.config,
        "--basedir",
        str(run_basedir),
        "--expname",
        expname,
        "--optimizer",
        args.optimizer,
        "--train_scheduler",
        args.train_scheduler,
        "--N_iters",
        str(args.n_iters),
        "--i_testset",
        str(args.i_testset if args.i_testset > 0 else args.n_iters),
        "--i_print",
        str(args.i_print),
        "--i_weights",
        str(args.i_weights),
        "--i_video",
        str(10**9),
        "--i_img",
        str(10**9),
        "--seed",
        str(args.seed + trial.trial_id),
    ]
    if args.deterministic:
        cmd.append("--deterministic")
    if not args.allow_reload:
        cmd.append("--no_reload")
    cmd.extend(params_to_cli(trial.params, space))
    cmd.extend(strip_remainder_marker(args.training_args))

    env = os.environ.copy()
    if cuda_device is not None and cuda_device != "":
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
    cwd = args.workdir or str(runner.parent)

    start = time.time()
    returncode = 999
    status = "fail"
    error = ""
    metric_value: Optional[float] = None
    metric_source = ""
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        with stdout_path.open("w", encoding="utf-8", errors="replace") as f:
            f.write("COMMAND: " + shlex.join(cmd) + "\n")
            f.write(f"CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', '')}\n")
            f.write(f"CWD={cwd}\n")
            f.write("=" * 120 + "\n")
            f.flush()
            proc = subprocess.run(
                cmd,
                cwd=cwd,
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout,
                check=False,
            )
        returncode = int(proc.returncode)
        metric_value, metric_source = parse_metric_from_files(exp_dir, stdout_path, args.metric)
        if returncode == 0 and metric_value is not None:
            status = "ok"
        elif returncode == 0:
            status = "metric_missing"
            error = f"Metric {args.metric!r} was not found. Expected readme.txt at {exp_dir / 'readme.txt'}"
        else:
            status = "returncode_nonzero"
            error = f"Training process returned {returncode}"
    except subprocess.TimeoutExpired as exc:
        status = "timeout"
        error = f"Timeout after {args.timeout_min} min: {exc}"
    except Exception as exc:  # pragma: no cover - robust orchestration
        status = "exception"
        error = repr(exc)

    duration = time.time() - start
    objective = objective_from_metric(metric_value, args.mode, args.failure_objective)
    record: Dict[str, Any] = {
        "trial_id": trial.trial_id,
        "phase": trial.phase,
        "status": status,
        "metric": args.metric,
        "metric_value": metric_value,
        "objective": objective,
        "returncode": returncode,
        "duration_sec": duration,
        "expname": expname,
        "exp_dir": str(exp_dir),
        "stdout_path": str(stdout_path),
        "metric_source": metric_source,
        "cuda_device": cuda_device,
        "x_unit": list(map(float, trial.x_unit)),
        "params": trial.params,
        "cmd": shlex.join(cmd),
        "error": error,
        "time_finished": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    append_jsonl(records_path, record)
    return record


def chunks(seq: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def evaluate_trials(
    trials: Sequence[TrialSpec],
    args: argparse.Namespace,
    space: Sequence[Dict[str, Any]],
    cuda_devices: Sequence[Optional[str]],
    run_basedir: Path,
    log_dir: Path,
    records_path: Path,
) -> List[Dict[str, Any]]:
    if not trials:
        return []
    max_parallel = max(1, len(cuda_devices))
    all_records: List[Dict[str, Any]] = []
    for batch in chunks(list(trials), max_parallel):
        with ThreadPoolExecutor(max_workers=len(batch)) as executor:
            future_to_trial = {}
            for idx, trial in enumerate(batch):
                cuda = cuda_devices[idx % len(cuda_devices)] if cuda_devices else None
                fut = executor.submit(run_one_trial, trial, args, space, cuda, run_basedir, log_dir, records_path)
                future_to_trial[fut] = trial
            for fut in as_completed(future_to_trial):
                rec = fut.result()
                all_records.append(rec)
                print(
                    f"[trial {rec['trial_id']:04d}] status={rec['status']} "
                    f"metric={rec.get('metric_value')} objective={rec.get('objective'):.6g} "
                    f"exp={rec['expname']}",
                    flush=True,
                )
    return all_records


def successful_training_tensors(records: Sequence[Dict[str, Any]], dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
    xs: List[List[float]] = []
    ys: List[List[float]] = []
    for rec in records:
        if rec.get("status") != "ok":
            continue
        x = rec.get("x_unit")
        y = rec.get("objective")
        if not isinstance(x, list) or len(x) != dim or y is None or not math.isfinite(float(y)):
            continue
        xs.append([float(v) for v in x])
        ys.append([float(y)])
    if not xs:
        return torch.empty(0, dim, dtype=torch.double), torch.empty(0, 1, dtype=torch.double)
    return torch.tensor(xs, dtype=torch.double), torch.tensor(ys, dtype=torch.double)


def propose_bo_candidates(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    q: int,
    raw_samples: int,
    num_restarts: int,
    mc_samples: int,
) -> torch.Tensor:
    dim = train_x.shape[-1]
    if train_x.shape[0] < max(3, dim + 1):
        # Too few observations for a stable GP. Fall back to Sobol.
        sobol = torch.quasirandom.SobolEngine(dimension=dim, scramble=True)
        return sobol.draw(q).to(dtype=torch.double)

    model = SingleTaskGP(
        train_X=train_x,
        train_Y=train_y,
        outcome_transform=Standardize(m=1),
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([mc_samples]))
    acq = qLogExpectedImprovement(
        model=model,
        best_f=train_y.max(),
        sampler=sampler,
    )
    bounds = torch.stack(
        [torch.zeros(dim, dtype=torch.double), torch.ones(dim, dtype=torch.double)]
    )
    candidates, _ = optimize_acqf(
        acq_function=acq,
        bounds=bounds,
        q=q,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        options={"batch_limit": 5, "maxiter": 200},
    )
    return candidates.detach().clamp(0.0, 1.0)


def best_record(records: Sequence[Dict[str, Any]], mode: str) -> Optional[Dict[str, Any]]:
    ok = [r for r in records if r.get("status") == "ok" and r.get("metric_value") is not None]
    if not ok:
        return None
    reverse = mode == "maximize"
    return sorted(ok, key=lambda r: float(r["metric_value"]), reverse=reverse)[0]


def main() -> None:
    args = parse_args()
    space = load_search_space(args.space_json)
    dim = len(space)

    hpo_dir = Path(args.hpo_dir).resolve()
    run_basedir = hpo_dir / "nerf_runs"
    log_dir = hpo_dir / "trial_logs"
    records_path = hpo_dir / "trials.jsonl"
    csv_path = hpo_dir / "trials.csv"
    hpo_dir.mkdir(parents=True, exist_ok=True)
    run_basedir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    cuda_devices: List[Optional[str]]
    if args.cuda_devices.strip() == "":
        cuda_devices = [None]
    else:
        cuda_devices = [x.strip() for x in args.cuda_devices.split(",") if x.strip() != ""]
    if not cuda_devices:
        cuda_devices = [None]

    records: List[Dict[str, Any]] = load_jsonl(records_path) if args.resume else []
    next_trial_id = 1 + max([int(r.get("trial_id", 0)) for r in records], default=0)

    print("Search space:")
    print(json.dumps(space, indent=2, ensure_ascii=False))
    print(f"HPO dir: {hpo_dir}")
    print(f"Metric: {args.metric} ({args.mode})")
    print(f"CUDA devices: {cuda_devices}")

    # Sobol initialization, accounting for already completed records on resume.
    num_existing = len(records)
    if num_existing < args.num_init:
        n_to_make = args.num_init - num_existing
        sobol = torch.quasirandom.SobolEngine(dimension=dim, scramble=True, seed=args.seed)
        # Draw enough points, then skip already-recorded initial IDs if resuming.
        x_all = sobol.draw(args.num_init).to(dtype=torch.double)
        init_trials: List[TrialSpec] = []
        for row in x_all[num_existing:]:
            x_unit = [float(v) for v in row.tolist()]
            params = unit_to_params(x_unit, space)
            init_trials.append(TrialSpec(next_trial_id, x_unit, params, "sobol"))
            next_trial_id += 1
        records.extend(evaluate_trials(init_trials, args, space, cuda_devices, run_basedir, log_dir, records_path))
        write_csv_summary(csv_path, records, space)

    for bo_iter in range(args.num_bo):
        train_x, train_y = successful_training_tensors(records, dim)
        if train_x.shape[0] == 0:
            raise RuntimeError("No successful trials are available; cannot fit BoTorch model.")

        q = max(1, int(args.q_batch))
        candidates = propose_bo_candidates(
            train_x=train_x,
            train_y=train_y,
            q=q,
            raw_samples=args.raw_samples,
            num_restarts=args.num_restarts,
            mc_samples=args.mc_samples,
        )
        bo_trials: List[TrialSpec] = []
        for row in candidates:
            x_unit = [float(v) for v in row.tolist()]
            params = unit_to_params(x_unit, space)
            bo_trials.append(TrialSpec(next_trial_id, x_unit, params, f"bo_{bo_iter:03d}"))
            next_trial_id += 1

        print(f"\n[BO iter {bo_iter + 1}/{args.num_bo}] proposing {len(bo_trials)} trial(s)")
        records.extend(evaluate_trials(bo_trials, args, space, cuda_devices, run_basedir, log_dir, records_path))
        write_csv_summary(csv_path, records, space)

        best = best_record(records, args.mode)
        if best is not None:
            print(
                f"Current best: trial={best['trial_id']} metric={best['metric_value']} "
                f"params={json.dumps(best['params'], ensure_ascii=False)}",
                flush=True,
            )

    write_csv_summary(csv_path, records, space)
    best = best_record(records, args.mode)
    best_path = hpo_dir / "best.json"
    if best is not None:
        best_path.write_text(json.dumps(best, indent=2, ensure_ascii=False), encoding="utf-8")
        print("\nBest trial:")
        print(json.dumps(best, indent=2, ensure_ascii=False))
        print(f"Saved best record to {best_path}")
    else:
        print("No successful trial found.")


if __name__ == "__main__":
    main()

"""Blender Lego
python botorch_nerf_tune.py \
  --runner ./run_nerf_ranksched.py \
  --config configs/lego.txt \
  --hpo-dir hpo_runs/lego_aux_auto_20trials_4gpu \
  --optimizer aux-sign-auto-cos-inc \
  --train-scheduler rank_wsd \
  --n-iters 100000 \
  --num-init 8 \
  --num-bo 3 \
  --q-batch 4 \
  --cuda-devices 0,1,2,3 \
  --metric testset_mean_psnr \
  --mode maximize \
  -- --dataset_type blender --white_bkgd --half_res

GPU2개 20trial
--num-init 8
--num-bo 6
--q-batch 2
--cuda-devices 0,1

"""