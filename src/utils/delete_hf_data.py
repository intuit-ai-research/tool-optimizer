import argparse
import logging
import os

from huggingface_hub import HfApi
from huggingface_hub.utils import EntryNotFoundError, HfHubHTTPError

log = logging.getLogger(__name__)


def delete_hf_data(
    hf_access_token: str | None = None,
    repo_id: str = "intuit/tool-optimizer-dataset",
    data_files: list[str] | None = None,
    data_dir: str | None = None,
) -> None:
    """Delete a file, list of files, or a directory from a HuggingFace dataset repo.
    
    Example Usage:
    ```bash
    cd src/utils
    export HF_TOKEN=<enter_hf_token>
    python delete_hf_data.py \
      --repo_id "intuit/tool-optimizer-dataset" \
      --data_dir "StableToolBench/tools_api"
    ```
    """
    token = hf_access_token or os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("No HuggingFace access token provided. Pass --hf_access_token or set the HF_TOKEN environment variable.")

    if not data_files and not data_dir:
        raise ValueError("Nothing to delete. Pass --data_files and/or --data_dir.")

    api = HfApi(token=token)

    repo_name = repo_id.split("/", 1)[1] if "/" in repo_id else repo_id

    def _normalize(path: str) -> str:
        clean = path.strip().strip("/")
        if clean.startswith(f"{repo_id}/"):
            clean = clean[len(repo_id) + 1 :]
            log.warning(f"Stripped repo_id prefix from path; using '{clean}'. Paths are relative to the repo root.")
        elif clean.startswith(f"{repo_name}/"):
            clean = clean[len(repo_name) + 1 :]
            log.warning(f"Stripped repo-name prefix from path; using '{clean}'. Paths are relative to the repo root.")
        return clean

    try:
        if data_files:
            for raw in data_files:
                file = _normalize(raw)
                log.info(f"Deleting file '{file}' from '{repo_id}'...")
                try:
                    api.delete_file(
                        path_in_repo=file,
                        repo_id=repo_id,
                        repo_type="dataset",
                        commit_message=f"Delete {file}",
                    )
                    log.info(f"Deleted '{file}'.")
                except EntryNotFoundError:
                    log.warning(f"'{file}' not found in '{repo_id}'. Skipping.")

        if data_dir:
            directory = _normalize(data_dir)
            log.info(f"Deleting directory '{directory}' from '{repo_id}'...")
            try:
                repo_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
            except HfHubHTTPError as e:
                log.error(f"Could not list repo files: {e}")
                raise
            prefix = f"{directory}/"
            if not any(f == directory or f.startswith(prefix) for f in repo_files):
                log.warning(f"No files found under '{directory}' in '{repo_id}'. Nothing to delete.")
            else:
                try:
                    api.delete_folder(
                        path_in_repo=directory,
                        repo_id=repo_id,
                        repo_type="dataset",
                        commit_message=f"Delete {directory}/",
                    )
                    log.info(f"Deleted directory '{directory}'.")
                except EntryNotFoundError:
                    log.warning(f"Directory '{directory}' not found in '{repo_id}'. Skipping.")

        log.info("Delete complete.")
    except HfHubHTTPError as e:
        log.error(f"Failed to delete data: {e}")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Delete files or directories from a HuggingFace dataset repo.")
    parser.add_argument("--hf_access_token", type=str, default=None, help="HuggingFace access token (defaults to HF_TOKEN env var)")
    parser.add_argument("--repo_id", type=str, default="intuit/tool-optimizer-dataset", help="HuggingFace dataset repo ID")
    parser.add_argument("--data_files", type=str, nargs="*", default=None, help="Specific files in the repo to delete")
    parser.add_argument("--data_dir", type=str, default=None, help="Directory in the repo to delete (recursive)")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    delete_hf_data(
        hf_access_token=args.hf_access_token,
        repo_id=args.repo_id,
        data_files=args.data_files,
        data_dir=args.data_dir,
    )


if __name__ == "__main__":
    main()
