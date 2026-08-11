from __future__ import annotations

import contextlib
import io
import json
import sys
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional

import boto3

from alpha_edge.core.audit import make_run_id, utc_now_iso, utc_today
from alpha_edge.core.runtime import runtime_engine_key
from alpha_edge.core.schemas import RuntimeConfig


@dataclass(frozen=True)
class ScriptRunLog:
    run_id: str
    script_name: str
    started_at_utc: str
    finished_at_utc: str
    status: str
    env: str
    bucket: str
    root: str
    argv: list[str]
    input_args: Dict[str, Any]
    stdout: str
    stderr: str
    error: Optional[str] = None


def script_log_key(cfg: RuntimeConfig, *, script_name: str, run_id: str, dt: Optional[str] = None) -> str:
    dt_norm = dt or utc_today()
    safe_script = script_name.replace("/", "_").replace(" ", "_")
    return runtime_engine_key(cfg, "script_logs", safe_script, f"dt={dt_norm}", f"{run_id}.json")


def write_script_run_log(
    *,
    cfg: RuntimeConfig,
    log: ScriptRunLog,
    dry_run: bool = False,
) -> str:
    key = script_log_key(cfg, script_name=log.script_name, run_id=log.run_id)

    if dry_run:
        print(f"[DRY RUN] Would write script log: s3://{cfg.bucket}/{key}")
        return key

    s3 = boto3.client("s3", region_name=cfg.region)
    s3.put_object(
        Bucket=cfg.bucket,
        Key=key,
        Body=json.dumps(asdict(log), indent=2, default=str).encode("utf-8"),
        ContentType="application/json",
    )
    print(f"[OK] Wrote script log: s3://{cfg.bucket}/{key}")
    return key


class Tee(io.TextIOBase):
    def __init__(self, original, buffer: io.StringIO):
        self.original = original
        self.buffer = buffer

    def write(self, s: str) -> int:
        self.buffer.write(s)
        return self.original.write(s)

    def flush(self) -> None:
        self.original.flush()


@contextlib.contextmanager
def capture_script_run(
    *,
    cfg: RuntimeConfig,
    script_name: str,
    input_args: Dict[str, Any],
    dry_run: bool = False,
    run_id: Optional[str] = None,
):
    """
    Capture stdout/stderr while still echoing to the terminal, then write a script log JSON to S3.

    Usage:
        with capture_script_run(cfg=cfg, script_name="record_trade.py", input_args=vars(args), dry_run=args.dry_run) as run_id:
            ... existing script body ...
    """
    rid = run_id or make_run_id()
    started = utc_now_iso()
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = Tee(old_stdout, stdout_buf)
    sys.stderr = Tee(old_stderr, stderr_buf)
    status = "success"
    error = None

    try:
        yield rid
    except Exception as exc:
        status = "failed"
        error = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        print(error, file=sys.stderr)
        raise
    finally:
        finished = utc_now_iso()
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        log = ScriptRunLog(
            run_id=rid,
            script_name=script_name,
            started_at_utc=started,
            finished_at_utc=finished,
            status=("dry_run" if dry_run and status == "success" else status),
            env=str(cfg.env),
            bucket=str(cfg.bucket),
            root=str(cfg.engine_root),
            argv=list(sys.argv),
            input_args=dict(input_args or {}),
            stdout=stdout_buf.getvalue(),
            stderr=stderr_buf.getvalue(),
            error=error,
        )
        write_script_run_log(cfg=cfg, log=log, dry_run=False if not dry_run else True)


def write_local_script_log(
    *,
    script_name: str,
    run_id: str,
    stdout: str,
    stderr: str,
    status: str,
    logs_root: str = "logs",
) -> str:
    dt = utc_today()
    safe_script = script_name.replace("/", "_").replace(" ", "_")
    path = Path(logs_root) / safe_script / f"dt={dt}" / f"{run_id}.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"status={status}\nrun_id={run_id}\nscript={script_name}\n\n--- stdout ---\n{stdout}\n\n--- stderr ---\n{stderr}\n",
        encoding="utf-8",
    )
    return str(path)
