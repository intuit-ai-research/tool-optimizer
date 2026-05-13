import argparse
import logging
import os

from huggingface_hub import hf_hub_download, snapshot_download

log = logging.getLogger(__name__)


def pull_hf_data(
    hf_access_token: str | None = None,
    repo_id: str = "intuit/tool-optimizer-dataset",
    data_files: list[str] | None = None,
    data_dir: str | None = None,
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
            log.info(f"Downloading directory '{data_dir}' from '{repo_id}'...")
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                allow_patterns=f"{data_dir}/*",
                local_dir=output_path,
                token=token,
            )

        if not data_files and not data_dir:
            log.info(f"Downloading entire dataset from '{repo_id}'...")
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=output_path,
                token=token,
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
    parser.add_argument("--data_dir", type=str, default=None, help="Specific directory to pull from the repo")
    parser.add_argument("--output_path", type=str, default="../../data", help="Local directory to save files to")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    pull_hf_data(output_path=args.output_path, hf_access_token=args.hf_access_token, repo_id=args.repo_id, data_files=args.data_files, data_dir=args.data_dir)


if __name__ == "__main__":
    main()
