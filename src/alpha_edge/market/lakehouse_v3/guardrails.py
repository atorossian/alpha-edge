"""Offline architecture guardrails for v3 market data-plane modules.

Only source-code files are inspected. This checker does not open market data,
access AWS or create local artifacts. Static rules complement (not replace)
IAM isolation, code review and behavioral tests.
"""

from __future__ import annotations

import ast
from pathlib import Path

from alpha_edge.market.lakehouse_v3.storage import LakehouseError

DATA_DIR = Path(__file__).resolve().parent

# No local operational market data even in temporary directories.
BANNED_IMPORTS = {"tempfile", "pickle", "sqlite3", "shelve"}
BANNED_CALLS = {
    "open",
    "builtins.open",
    "os.open",
    "os.replace",
    "os.rename",
    "os.remove",
    "os.unlink",
    "pd.read_csv",
    "pd.read_parquet",
    "pandas.read_csv",
    "pandas.read_parquet",
    "Path.write_bytes",
    "Path.write_text",
    "Path.open",
}
BANNED_METHODS = {
    "to_csv",
    "to_pickle",
    "to_json",
    "to_excel",
    "write_bytes",
    "write_text",
    "mkdir",
    "mkdtemp",
    "NamedTemporaryFile",
    "TemporaryDirectory",
}
# The in-memory bootstrap reader and in-memory S3 parquet reader are
# explicit exceptions and must be reviewed on edits.
IN_MEMORY_READ_EXCEPTIONS = {
    ("bootstrap.py", "pd.read_csv"),
    ("storage.py", "pd.read_parquet"),
}


def _name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return _name(node.value) + "." + node.attr
    return ""


def lint_text(source: str, filename: str) -> list[str]:
    """Pure code-string analysis; no filesystem I/O."""
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as exc:
        return [f"{filename}:{exc.lineno}: invalid Python syntax"]
    problems: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in BANNED_IMPORTS:
                    problems.append(f"{filename}:{node.lineno}: forbidden import {alias.name}")
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module.split(".")[0] in BANNED_IMPORTS:
                problems.append(f"{filename}:{node.lineno}: forbidden import {node.module}")
        elif isinstance(node, ast.Call):
            call = _name(node.func)
            method = call.split(".")[-1]
            if (filename, call) in IN_MEMORY_READ_EXCEPTIONS:
                # S3 byte streams via BytesIO; do not permit new paths.
                if not node.args or _name(node.args[0].func) != "io.BytesIO" if isinstance(node.args[0], ast.Call) else True:
                    problems.append(f"{filename}:{node.lineno}: in-memory read exception must use io.BytesIO")
                continue
            if call in BANNED_CALLS or method in BANNED_METHODS:
                problems.append(f"{filename}:{node.lineno}: forbidden local I/O call {call}")
            # pandas/Arrow Parquet serialization must write to a BytesIO,
            # never a string or Path.
            if method == "to_parquet":
                if filename != "storage.py" or not node.args or _name(node.args[0]) != "stream":
                    problems.append(f"{filename}:{node.lineno}: Parquet must be serialized to BytesIO")
    return problems


def validate_source_tree(files: dict[str, str]) -> None:
    problems = []
    for name, body in files.items():
        problems.extend(lint_text(body, name))
    if problems:
        raise LakehouseError("Local market-data I/O guard failed:\n" + "\n".join(problems))


def main() -> None:
    scripts = {
        path.name: path.read_text(encoding="utf-8")
        for path in DATA_DIR.glob("*.py")
        if path.name not in {"guardrails.py"}
    }
    validate_source_tree(scripts)
    print(f"S3-only source-code I/O guard passed for {len(scripts)} modules")


if __name__ == "__main__":
    main()
