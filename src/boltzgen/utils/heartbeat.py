import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pytorch_lightning as pl


class HeartbeatCallback(pl.Callback):  # type: ignore[attr-defined]
    """Periodically writes a JSON heartbeat with step progress and ETA.

    Notes
    -----
    - Writes to `<step_output_dir>/status.json` (atomically via tmp+rename)
    - Only the global rank (rank 0) writes the file
    - Percent/ETA are computed from on-disk artifact counts with a baseline
    - Expected totals are derived from the datamodule and model predict args
    """

    def __init__(
        self,
        *,
        output_dir: str,
        writer: object,
        root_dir: Optional[str] = None,
        interval_seconds: Optional[float] = None,
        throughput_window_seconds: float = 60.0,
    ) -> None:
        super().__init__()
        self.output_dir = Path(output_dir)
        self.root_dir = Path(root_dir) if root_dir is not None else None
        self.writer = writer
        self.interval_seconds = (
            float(os.environ.get("BOLTZGEN_HEARTBEAT_INTERVAL", "0"))
            if interval_seconds is None
            else float(interval_seconds)
        )
        if self.interval_seconds <= 0:
            self.interval_seconds = 5.0
        self.throughput_window_seconds = throughput_window_seconds

        # Runtime state
        self._last_write_ts: float = 0.0
        self._baseline_counts: Dict[str, int] = {}
        self._primary_counter: str = ""
        self._fixed_expected_total: Optional[int] = None
        self._recent: List[Tuple[float, int]] = []  # (timestamp, produced_since_start)
        self._start_ts: float = 0.0

    # --------------- Lightning Hooks ---------------
    def on_predict_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:  # type: ignore[override]
        if not getattr(trainer, "is_global_zero", True):
            return
        self._start_ts = time.time()

        # Decide primary counter by writer type/attributes
        counters = self._discover_counters()
        # Choose the first counter as primary; this maps to the step's main artifact
        if counters:
            self._primary_counter = counters[0]["name"]
        else:
            self._primary_counter = "outputs"

        # Baseline counts (for resume/reuse) and initial write
        self._baseline_counts = {c["name"]: self._count_files(Path(c["dir"]), c["pattern"], c.get("exclude")) for c in counters}
        self._recent = [(self._start_ts, 0)]

        # For design, compute a fixed expected_total from cfg.multiplicity and diffusion_samples
        try:
            writer_type = type(self.writer).__name__
            if writer_type == "DesignWriter":
                cfg = getattr(trainer.datamodule, "cfg", None)
                if cfg is not None:
                    multiplicity = int(getattr(cfg, "multiplicity", 0) or 0)
                    yaml_path = getattr(cfg, "yaml_path", None)
                    if isinstance(yaml_path, list):
                        num_specs = len(yaml_path)
                    else:
                        num_specs = 1 if yaml_path is not None else 0
                    ds = int(getattr(pl_module, "predict_args", {}).get("diffusion_samples", 1) or 1)
                    if multiplicity and num_specs and ds:
                        self._fixed_expected_total = multiplicity * ds * num_specs
        except Exception:
            self._fixed_expected_total = self._fixed_expected_total or None

        self._write_heartbeat(trainer, pl_module, force=True)

    def on_predict_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:  # type: ignore[override]
        if not getattr(trainer, "is_global_zero", True):
            return
        self._write_heartbeat(trainer, pl_module)

    def on_predict_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:  # type: ignore[override]
        if not getattr(trainer, "is_global_zero", True):
            return
        self._write_heartbeat(trainer, pl_module, mark_complete=True, force=True)

    # --------------- Internal ---------------
    def _discover_counters(self) -> List[Dict[str, str]]:
        """Return a list of counters with name, dir, pattern, exclude."""
        counters: List[Dict[str, str]] = []
        # Common case: DesignWriter writes CIFs directly to writer.outdir
        outdir = getattr(self.writer, "outdir", None)
        if isinstance(outdir, (str, Path)) and outdir:
            outdir = str(outdir)
            # Heuristics based on directory name for readable names
            base = Path(outdir).name
            if base == "intermediate_designs":
                counters.append({"name": "design_cif", "dir": outdir, "pattern": "*.cif", "exclude": "*_native.cif"})
            elif base == "intermediate_designs_inverse_folded":
                counters.append({"name": "inverse_fold_cif", "dir": outdir, "pattern": "*.cif", "exclude": "*_native.cif"})
            elif base in {"fold_out_npz", "fold_out_design_npz", "affinity_out_npz"}:
                # Folding or affinity primary artifacts are npz
                name = (
                    "fold_npz" if base.startswith("fold_out") else "affinity_npz"
                )
                counters.append({"name": name, "dir": outdir, "pattern": "*.npz"})
            else:
                # Default to counting CIFs for unknown design-like dirs
                counters.append({"name": f"{base}_cif", "dir": outdir, "pattern": "*.cif", "exclude": "*_native.cif"})

        # FoldingWriter exposes a refold CIF dir as well; include as secondary counter
        refold_dir = getattr(self.writer, "refold_cif_dir", None)
        if isinstance(refold_dir, (str, Path)) and refold_dir:
            counters.append({"name": "refold_cif", "dir": str(refold_dir), "pattern": "*.cif"})

        return counters

    def _count_files(self, directory: Path, pattern: str, exclude: Optional[str] = None) -> int:
        if not directory.exists():
            return 0
        matched = list(directory.glob(pattern))
        if exclude:
            matched = [p for p in matched if not p.match(exclude)]
        return len(matched)

    def _latest_file(self, directory: Path, pattern: str, exclude: Optional[str] = None) -> Optional[Path]:
        if not directory.exists():
            return None
        files = list(directory.glob(pattern))
        if exclude:
            files = [p for p in files if not p.match(exclude)]
        if not files:
            return None
        return max(files, key=lambda p: p.stat().st_mtime)

    def _write_heartbeat(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        mark_complete: bool = False,
        force: bool = False,
    ) -> None:
        now = time.time()
        if not force and (now - self._last_write_ts) < self.interval_seconds:
            return

        counters = self._discover_counters()
        counts_now: Dict[str, int] = {}
        latest_path: Optional[Path] = None
        latest_time: Optional[float] = None

        for c in counters:
            d = Path(c["dir"])
            cnt = self._count_files(d, c["pattern"], c.get("exclude"))
            counts_now[c["name"]] = cnt
            latest = self._latest_file(d, c["pattern"], c.get("exclude"))
            if latest is not None:
                ts = latest.stat().st_mtime
                if latest_time is None or ts > latest_time:
                    latest_time = ts
                    latest_path = latest

        # Produced totals
        produced_total = counts_now.get(self._primary_counter, 0)  # absolute
        produced_since_start = produced_total - self._baseline_counts.get(self._primary_counter, 0)  # relative to this run
        if produced_since_start < 0:
            produced_since_start = 0

        # Update throughput window
        self._recent.append((now, produced_since_start))
        cutoff = now - self.throughput_window_seconds
        while len(self._recent) > 1 and self._recent[0][0] < cutoff:
            self._recent.pop(0)
        # Compute rate (items/sec) over the window
        if len(self._recent) >= 2:
            dt = max(self._recent[-1][0] - self._recent[0][0], 1e-6)
            dd = max(self._recent[-1][1] - self._recent[0][1], 0)
            rate_per_sec = dd / dt
        else:
            rate_per_sec = 0.0

        # Percent and ETA (absolute progress)
        if self._fixed_expected_total:
            expected = max(int(self._fixed_expected_total), 1)
        else:
            # Remaining items come from current datamodule length; absolute expected = already produced + remaining
            try:
                remaining_items = len(getattr(trainer.datamodule, "predict_set"))
            except Exception:
                remaining_items = 0
            writer_type = type(self.writer).__name__
            # Only DesignWriter creates multiple artifacts per item (diffusion_samples). Others write one per item.
            if writer_type == "DesignWriter":
                try:
                    ds = int(getattr(pl_module, "predict_args", {}).get("diffusion_samples", 1))
                except Exception:
                    ds = 1
                expected = produced_total + remaining_items * max(1, ds)
            else:
                expected = produced_total + remaining_items

        expected = max(expected, 1)
        percent = min(max(produced_total / expected, 0.0), 1.0)
        remaining = max(expected - produced_total, 0)
        eta_seconds: Optional[float] = None
        if mark_complete:
            percent = 1.0
            eta_seconds = 0.0
        elif rate_per_sec > 0:
            eta_seconds = remaining / rate_per_sec
        else:
            # Fallback to overall average since start if window shows 0
            elapsed = max(now - self._start_ts, 1e-6)
            overall_rate = produced_since_start / elapsed
            if overall_rate > 0:
                eta_seconds = remaining / overall_rate

        # Step context from env
        step_label = os.environ.get("BOLTZGEN_PIPELINE_PROGRESS", "")
        step_name = os.environ.get("BOLTZGEN_PIPELINE_STEP", "")
        step_index: Optional[int] = None
        num_steps: Optional[int] = None
        if step_label:
            # Expect formats like "Step i/N"
            import re

            m = re.search(r"(\d+)\/(\d+)", step_label)
            if m:
                try:
                    step_index = int(m.group(1))
                    num_steps = int(m.group(2))
                except Exception:
                    step_index = None
                    num_steps = None

        # Compute device info
        world_size = getattr(trainer, "world_size", None)
        if world_size is None:
            # Fallback approximation
            try:
                world_size = max(1, int(getattr(trainer, "num_devices", 1)))
            except Exception:
                world_size = 1

        compute_info: Dict[str, object] = {
            "devices": int(getattr(trainer, "num_devices", 1) or 1),
            "nodes": int(getattr(trainer, "num_nodes", 1) or 1),
            "precision": getattr(trainer, "_precision", None) or getattr(trainer, "precision", None),
            "global_rank": int(getattr(trainer, "global_rank", 0) or 0),
            "world_size": int(world_size or 1),
        }

        # Build counters section
        counters_out: List[Dict[str, object]] = []
        for c in counters:
            name = c["name"]
            base = self._baseline_counts.get(name, 0)
            current = counts_now.get(name, 0)
            counters_out.append(
                {
                    "name": name,
                    "dir": c["dir"],
                    "count": current,
                    "since_start": max(current - base, 0),
                    "pattern": c["pattern"],
                    **({"exclude": c["exclude"]} if c.get("exclude") else {}),
                }
            )

        # Writer info (if available)
        writer_failed = int(getattr(self.writer, "failed", 0) or 0)
        writer_type = type(self.writer).__name__

        status: Dict[str, object] = {
            "job": {
                "output_dir": str(self.output_dir.resolve()),
                "pid": os.getpid(),
            },
            "pipeline": {
                "step_name": step_name,
                "step_index": step_index,
                "num_steps": num_steps,
                "label": (f"Step {step_index}/{num_steps}: {step_name}" if step_index and num_steps and step_name else step_label or step_name),
            },
            "status": {
                "state": "completed" if mark_complete else "running",
                "started_at": self._iso(self._start_ts),
                "updated_at": self._iso(now),
                "percent": percent,
                "eta_seconds": eta_seconds,
                "eta_timestamp": self._iso(now + eta_seconds) if isinstance(eta_seconds, (int, float)) else None,
            },
            "compute": compute_info,
            "progress": {
                "expected_total": expected,
                "produced_total": produced_total,
                "produced_since_start": produced_since_start,
                "throughput_per_min": rate_per_sec * 60.0,
                "throughput_window_sec": self.throughput_window_seconds,
                "primary_counter": self._primary_counter,
                "counters": counters_out,
            },
            "writer": {
                "type": writer_type,
                "failed": writer_failed,
                "last_file": str(latest_path) if latest_path else None,
                "last_write_time": self._iso(latest_time) if latest_time else None,
            },
        }

        # Write to step directory
        self._atomic_write_json(self.output_dir / "status.json", status)
        # Also write to run root, if provided
        if self.root_dir is not None:
            self._atomic_write_json(self.root_dir / "status.json", status)
        self._last_write_ts = now

    @staticmethod
    def _iso(ts: Optional[float]) -> Optional[str]:
        if ts is None:
            return None
        try:
            return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))
        except Exception:
            return None

    @staticmethod
    def _atomic_write_json(path: Path, data: Dict[str, object]) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")))
            tmp.replace(path)
        except Exception:
            # Best-effort only; never crash the run on heartbeat failure
            pass


