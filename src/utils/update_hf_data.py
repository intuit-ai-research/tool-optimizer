import argparse
import logging
import math
import os
from collections import defaultdict
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi
from huggingface_hub.utils import EntryNotFoundError, HfHubHTTPError

log = logging.getLogger(__name__)

# Files >= this size are uploaded in their own commit so a timeout on a big file
# doesn't waste a batch of small ones that already finished uploading.
LARGE_FILE_THRESHOLD_BYTES = 200 * 1024 * 1024  # 200 MiB
# HF rejects commits past a few hundred ops; cap well below to be safe.
MAX_FILES_PER_COMMIT = 200
# Soft cap on total bytes per commit so a single batch doesn't run for too long.
MAX_BYTES_PER_COMMIT = 2 * 1024 * 1024 * 1024  # 2 GiB


def _iter_local_files(folder_path: str, include_hidden: bool = False) -> list[tuple[Path, str]]:
    """Return (absolute_path, posix_relative_path) tuples for every file under folder_path.

    By default, skips hidden files and any path that has a hidden ancestor directory
    (anything starting with '.'). Notably this excludes '.cache/huggingface/', which the
    HF client creates locally for upload tracking and which would otherwise be uploaded
    alongside real data.
    """
    root = Path(folder_path).resolve()
    if not root.is_dir():
        raise ValueError(f"'{folder_path}' is not a directory.")
    files: list[tuple[Path, str]] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if not include_hidden and any(part.startswith(".") for part in rel.parts):
            continue
        files.append((path, rel.as_posix()))
    files.sort(key=lambda f: f[1])
    return files


def _build_remote_index(api: HfApi, repo_id: str, path_in_repo: str) -> dict[str, int]:
    """Return {repo_relative_path: size_in_bytes} for files already in the repo under path_in_repo."""
    try:
        repo_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    except (EntryNotFoundError, HfHubHTTPError) as e:
        log.warning(f"Could not list remote files (treating repo as empty): {e}")
        return {}

    prefix = path_in_repo.strip("/")
    if prefix:
        scoped = [f for f in repo_files if f.startswith(f"{prefix}/")]
    else:
        scoped = list(repo_files)
    if not scoped:
        return {}

    try:
        infos = api.get_paths_info(repo_id=repo_id, paths=scoped, repo_type="dataset", expand=False)
    except HfHubHTTPError as e:
        log.warning(f"Could not fetch remote file sizes (will re-upload to be safe): {e}")
        return {}

    index: dict[str, int] = {}
    for info in infos:
        size = getattr(info, "size", None)
        if size is None and getattr(info, "lfs", None) is not None:
            size = getattr(info.lfs, "size", None)
        if size is not None:
            index[info.path] = int(size)
    return index


def update_hf_data(
    hf_access_token: str | None = None,
    repo_id: str = "intuit/tool-optimizer-dataset",
    input_data_path: str = "data/",
    path_in_repo: str = "",
    max_commits: int = 20,
    include_hidden: bool = False,
) -> None:
    """Upload a local folder to a HuggingFace dataset repo under path_in_repo, skipping files already present.

    Files are grouped by their parent directory and committed together — a directory of small files becomes
    a single commit. Directories that exceed the per-commit caps (MAX_FILES_PER_COMMIT,
    MAX_BYTES_PER_COMMIT) are auto-chunked. If the directory-grouped plan would exceed max_commits, the
    script falls back to flat packing (~ceil(small_file_count / max_commits) files per commit, ignoring
    directory boundaries) so the run stays under HF's 128 commits/hour free-tier limit.

    Files >=LARGE_FILE_THRESHOLD_BYTES are always committed individually so a timeout on one big file
    doesn't waste a batch. They are not counted against max_commits (HF requires LFS uploads to commit
    individually).

    The local folder's basename becomes the top-level subdir under path_in_repo. For example, uploading
    "foo/" with path_in_repo="x/y" to "intuit/tool-optimizer-dataset" places files at
    "intuit/tool-optimizer-dataset/x/y/foo/...".

    Example Usage:
    ```bash
    cd src/utils
    export HF_TOKEN=<enter_hf_token>
    python update_hf_data.py \
      --repo_id "intuit/tool-optimizer-dataset" \
      --input_data_path ../../data/StableToolBench/tools_api \
      --path_in_repo "StableToolBench/ \
      --max_commits 10"
    ```
    """
    token = hf_access_token or os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("No HuggingFace access token provided. Pass --hf_access_token or set the HF_TOKEN environment variable.")

    api = HfApi(token=token)

    local_root = Path(input_data_path).resolve()
    folder_name = local_root.name
    repo_prefix = "/".join(p for p in [path_in_repo.strip("/"), folder_name] if p)

    log.info(f"Scanning '{local_root}'...")
    local_files = _iter_local_files(str(local_root), include_hidden=include_hidden)
    total = len(local_files)
    if total == 0:
        log.info("No files found to upload.")
        return
    hidden_note = "" if include_hidden else " (hidden files/dirs excluded; pass --include_hidden to keep them)"
    log.info(f"Found {total} local files{hidden_note}. Destination: '{repo_id}/{repo_prefix}/'")

    log.info(f"Fetching remote file index under '{repo_prefix}/'...")
    remote_index = _build_remote_index(api, repo_id, repo_prefix)
    log.info(f"Found {len(remote_index)} existing remote files under '{repo_prefix}/'.")

    pending: list[tuple[Path, str, int]] = []
    skipped = 0
    for abs_path, rel_path in local_files:
        repo_path = f"{repo_prefix}/{rel_path}" if repo_prefix else rel_path
        local_size = abs_path.stat().st_size
        if remote_index.get(repo_path) == local_size:
            skipped += 1
            continue
        pending.append((abs_path, repo_path, local_size))

    remaining = len(pending)
    log.info(f"Skipping {skipped} already-uploaded files. {remaining} files to upload.")
    if remaining == 0:
        log.info("Nothing to do. Upload complete.")
        return

    uploaded = 0
    failed: list[str] = []

    def _commit(batch: list[tuple[Path, str, int]], message: str) -> None:
        nonlocal uploaded
        operations = [
            CommitOperationAdd(path_in_repo=repo_path, path_or_fileobj=str(abs_path)) for abs_path, repo_path, _ in batch
        ]
        api.create_commit(
            repo_id=repo_id,
            repo_type="dataset",
            operations=operations,
            commit_message=message,
        )
        uploaded += len(batch)
        for _, repo_path, _ in batch:
            log.info(f"[{uploaded + skipped}/{total}] uploaded {repo_path}")

    def _flush(batch: list[tuple[Path, str, int]], label: str) -> None:
        if not batch:
            return
        try:
            _commit(batch, f"Upload {len(batch)} files to {label}")
        except Exception as e:
            log.error(f"Batch commit failed ({len(batch)} files under {label}): {e}")
            failed.extend(p for _, p, _ in batch)

    # Split pending into small files (batched) and large files (always solo commits).
    small_files: list[tuple[Path, str, int]] = []
    large_files: list[tuple[Path, str, int]] = []
    for abs_path, repo_path, size in pending:
        if size >= LARGE_FILE_THRESHOLD_BYTES:
            large_files.append((abs_path, repo_path, size))
        else:
            small_files.append((abs_path, repo_path, size))

    def _flat_pack(files: list[tuple[Path, str, int]], commit_budget: int, label: str) -> None:
        target_per_commit = math.ceil(len(files) / commit_budget)
        batch: list[tuple[Path, str, int]] = []
        batch_bytes = 0
        for abs_path, repo_path, size in files:
            would_exceed = len(batch) >= target_per_commit or batch_bytes + size > MAX_BYTES_PER_COMMIT
            if batch and would_exceed:
                _flush(batch, label)
                batch = []
                batch_bytes = 0
            batch.append((abs_path, repo_path, size))
            batch_bytes += size
        _flush(batch, label)

    label_root = f"{repo_prefix}/" if repo_prefix else "<root>"

    if small_files:
        by_dir: dict[str, list[tuple[Path, str, int]]] = defaultdict(list)
        for abs_path, repo_path, size in small_files:
            parent = repo_path.rsplit("/", 1)[0] if "/" in repo_path else ""
            by_dir[parent].append((abs_path, repo_path, size))

        if len(by_dir) > max_commits:
            log.info(
                f"{len(by_dir)} subdirectories exceeds max_commits={max_commits}. "
                f"Packing {len(small_files)} small files across directories "
                f"(~{math.ceil(len(small_files) / max_commits)} per commit)."
            )
            _flat_pack(small_files, max_commits, label_root)
        else:
            for directory in sorted(by_dir.keys()):
                files = by_dir[directory]
                label = f"{directory}/" if directory else f"{repo_prefix}/"
                batch: list[tuple[Path, str, int]] = []
                batch_bytes = 0
                for abs_path, repo_path, size in files:
                    would_exceed = len(batch) >= MAX_FILES_PER_COMMIT or batch_bytes + size > MAX_BYTES_PER_COMMIT
                    if batch and would_exceed:
                        _flush(batch, label)
                        batch = []
                        batch_bytes = 0
                    batch.append((abs_path, repo_path, size))
                    batch_bytes += size
                _flush(batch, label)

    for abs_path, repo_path, size in large_files:
        try:
            _commit([(abs_path, repo_path, size)], f"Upload {repo_path}")
        except Exception as e:
            log.error(f"Failed to upload large file {repo_path}: {e}")
            failed.append(repo_path)

    log.info(f"Done. Uploaded {uploaded}/{remaining} pending files (skipped {skipped} already-present, {len(failed)} failed).")
    if failed:
        log.error(f"{len(failed)} files failed to upload. Re-run the same command to retry only those files.")
        for p in failed[:20]:
            log.error(f"  failed: {p}")
        if len(failed) > 20:
            log.error(f"  ... and {len(failed) - 20} more")
        raise RuntimeError(f"{len(failed)} files failed to upload")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync local data folder to a HuggingFace dataset repo.")
    parser.add_argument("--hf_access_token", type=str, default=None, help="HuggingFace access token (defaults to HF_TOKEN env var)")
    parser.add_argument("--repo_id", type=str, default="intuit/tool-optimizer-dataset", help="HuggingFace dataset repo ID")
    parser.add_argument("--input_data_path", type=str, default="../../data/", help="Local folder to upload")
    parser.add_argument(
        "--path_in_repo",
        type=str,
        default="",
        help="Subdirectory in the repo to upload into. The local folder's basename is appended (e.g. path_in_repo='x/y' + folder 'foo' -> 'x/y/foo').",
    )
    parser.add_argument(
        "--max_commits",
        type=int,
        default=20,
        help="Cap the number of commits for small files (HF free-tier limit is 128/hour). If the directory-grouped plan would exceed this, files are flat-packed into ~ceil(N/max_commits) per commit. Large files (>=200MiB) always commit individually and don't count against this budget.",
    )
    parser.add_argument(
        "--include_hidden",
        action="store_true",
        help="Include hidden files and directories (anything starting with '.'). Excluded by default to avoid uploading the HF client's local '.cache/huggingface/' tracking directory.",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    update_hf_data(
        hf_access_token=args.hf_access_token,
        repo_id=args.repo_id,
        input_data_path=args.input_data_path,
        path_in_repo=args.path_in_repo,
        max_commits=args.max_commits,
        include_hidden=args.include_hidden,
    )


if __name__ == "__main__":
    main()
