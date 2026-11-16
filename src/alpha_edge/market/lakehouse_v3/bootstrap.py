"""S3-only bootstrap from the five current universe configuration files.

GitHub provides a pinned initial config revision. Runtime consumers read S3.
No historical market-data lake, past backfills, or local file is consulted.
"""

from __future__ import annotations

import argparse
import json
import re
from urllib.request import urlopen

from alpha_edge.market.lakehouse_v3.registry_v3 import SOURCE_FILES, stage_registry

_REPO = "atorossian/alpha-edge"
_SHA = re.compile(r"^[0-9a-f]{40}$")


def fetch_five_github_files(commit_sha: str, *, opener=urlopen) -> dict[str, bytes]:
    if not _SHA.fullmatch(commit_sha):
        raise ValueError("Bootstrap source must be a full 40-character Git commit SHA")
    contents = {}
    for name in SOURCE_FILES:
        url = f"https://raw.githubusercontent.com/{_REPO}/{commit_sha}/data/universe/{name}"
        with opener(url, timeout=30) as response:
            data = response.read(8 * 1024 * 1024 + 1)
        if len(data) > 8 * 1024 * 1024:
            raise ValueError(f"Unexpectedly large configuration source: {name}")
        contents[name] = data
    return contents


def bootstrap(s3, bucket: str, *, source_commit: str, execute: bool = False) -> dict:
    files = fetch_five_github_files(source_commit)
    result = stage_registry(s3, bucket, files, execute=execute)
    result["github_repo"] = _REPO
    result["github_commit"] = source_commit
    result["source_file_names"] = list(SOURCE_FILES)
    return result


def main() -> None:
    p = argparse.ArgumentParser(description="Canonical S3 registry from five current universe files")
    p.add_argument("--bucket", default="alpha-edge-algo")
    p.add_argument("--source-commit", required=True, help="40-character immutable Git commit SHA")
    p.add_argument("--execute", action="store_true", help="Stage six immutable S3 objects, never activate")
    args = p.parse_args()
    import boto3
    result = bootstrap(
        boto3.client("s3", region_name="eu-west-1"),
        args.bucket, source_commit=args.source_commit, execute=args.execute,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
