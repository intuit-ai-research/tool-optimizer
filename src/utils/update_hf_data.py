import argparse
import logging
import os

from huggingface_hub import HfApi

log = logging.getLogger(__name__)


def update_hf_data(
    hf_token: str | None = None, repo_id: str = "intuit/tool-optimizer-dataset", input_data_path: str = "data/"
) -> None:
    """Upload local data folder to a HuggingFace dataset repo, pushing only the diff."""
    token = hf_token or os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("No HuggingFace token provided. Pass --hf_token or set the HF_TOKEN environment variable.")

    try:
        api = HfApi(token=token)

        log.info(f"Uploading '{input_data_path}' to '{repo_id}'...")

        api.upload_folder(
            input_data_path=input_data_path,
            repo_id=repo_id,
            repo_type="dataset",
            delete_patterns="*",
        )

        log.info("Upload complete.")
    except Exception as e:
        log.error(f"Failed to upload data: {e}")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync local data folder to a HuggingFace dataset repo.")
    parser.add_argument("--hf_token", type=str, default=None, help="HuggingFace access token (defaults to HF_TOKEN env var)")
    parser.add_argument("--repo_id", type=str, default="intuit/tool-optimizer-dataset", help="HuggingFace dataset repo ID")
    parser.add_argument("--input_data_path", type=str, default="data/", help="Local folder to upload")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    update_hf_data(hf_token=args.hf_token, repo_id=args.repo_id, input_data_path=args.input_data_path)


if __name__ == "__main__":
    main()
