import argparse
import logging
import os
import time

from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.utils import HfHubHTTPError

log = logging.getLogger(__name__)

# HF Hub free tier: 5000 resolver requests per 5 minutes.
# Re-downloads of cached files don't count, so retries become cheap on subsequent attempts.
RATE_LIMIT_MAX_RETRIES = 6
RATE_LIMIT_BACKOFF_SECONDS = 60
DOWNLOAD_WORKERS = 2


def _snapshot_with_retry(**kwargs) -> None:
    for attempt in range(RATE_LIMIT_MAX_RETRIES):
        try:
            snapshot_download(**kwargs)
            return
        except HfHubHTTPError as e:
            status = getattr(e.response, "status_code", None)
            if status != 429 or attempt == RATE_LIMIT_MAX_RETRIES - 1:
                raise
            wait = RATE_LIMIT_BACKOFF_SECONDS * (2**attempt)
            log.warning(f"Hit HF Hub rate limit (429). Sleeping {wait}s before retry {attempt + 2}/{RATE_LIMIT_MAX_RETRIES}. Already-cached files will not be re-resolved.")
            time.sleep(wait)


def pull_hf_data(
    hf_access_token: str | None = None,
    repo_id: str = "intuit/tool-optimizer-dataset",
    data_files: list[str] | None = None,
    data_dir: list[str] | None = None,
    output_path: str = "../../data",
) -> None:
    """Pull data files from a HuggingFace dataset repo."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.normpath(os.path.join(script_dir, output_path))

    if not os.path.exists(output_path):
        log.info(f"Output path '{output_path}' does not exist. Creating it.")
        os.makedirs(output_path, exist_ok=True)

    token = hf_access_token or os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("No HuggingFace access token provided. Pass --hf_access_token or set the HF_TOKEN environment variable.")

    try:
        if data_files:
            for file in data_files:
                log.info(f"Downloading '{file}' from '{repo_id}'...")
                hf_hub_download(
                    repo_id=repo_id,
                    repo_type="dataset",
                    filename=file,
                    local_dir=output_path,
                    token=token,
                )

        if data_dir:
            log.info(f"Downloading directories {data_dir} from '{repo_id}'...")
            _snapshot_with_retry(
                repo_id=repo_id,
                repo_type="dataset",
                allow_patterns=[f"{d}/*" for d in data_dir],
                local_dir=output_path,
                token=token,
                max_workers=DOWNLOAD_WORKERS,
            )

        if not data_files and not data_dir:
            log.info(f"Downloading entire dataset from '{repo_id}'...")
            _snapshot_with_retry(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=output_path,
                token=token,
                max_workers=DOWNLOAD_WORKERS,
            )

        log.info(f"Download complete. Files saved to '{output_path}'.")
    except Exception as e:
        log.error(f"Failed to pull data: {e}")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Pull data files from a HuggingFace dataset repo.")
    parser.add_argument("--hf_access_token", type=str, default=None, help="HuggingFace access token (defaults to HF_TOKEN env var)")
    parser.add_argument("--repo_id", type=str, default="intuit/tool-optimizer-dataset", help="HuggingFace dataset repo ID")
    parser.add_argument("--data_files", type=str, nargs="*", default=None, help="Specific files to pull (pulls entire repo if omitted)")
    parser.add_argument("--data_dir", type=str, nargs="*", default=None, help="One or more directories to pull from the repo (space-separated)")
    parser.add_argument("--output_path", type=str, default="../../data", help="Local directory to save files to")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    pull_hf_data(output_path=args.output_path, hf_access_token=args.hf_access_token, repo_id=args.repo_id, data_files=args.data_files, data_dir=args.data_dir)


if __name__ == "__main__":
    main()
