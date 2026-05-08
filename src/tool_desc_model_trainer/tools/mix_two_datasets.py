# mix two datasets (folders) into per-epoch mixed datasets
import argparse
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mix two datasets per epoch with a given ratio.")
    parser.add_argument(
        "--dataset_folder1",
        type=str,
        default="/home/sagemaker-user/user-default-efs/FunctionWrapper_PL/data_stb/split2b_w_eval1",
        help="Folder treated as the w_eval split (provides mix_ratio share each epoch).",
    )
    parser.add_argument(
        "--dataset_folder2",
        type=str,
        default="/home/sagemaker-user/user-default-efs/FunctionWrapper_PL/data_stb/split2b_no_eval1",
        help="Second folder (provides 1 - mix_ratio share each epoch).",
    )
    parser.add_argument(
        "--mix_ratios",
        type=float,
        nargs="+",
        default=[0.5],
        help="List of per-epoch fractions from dataset_folder1 (length must equal num_epochs).",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1,
        help="How many mixed epochs to generate (one folder per epoch).",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="/home/sagemaker-user/user-default-efs/FunctionWrapper_PL/data_stb_mixed",
        help="Root directory where split2b_mix_* folders will be written.",
    )
    parser.add_argument(
        "--total_train_size",
        type=int,
        nargs="+",
        default=None,
        help="Total train rows per epoch. Single value applies to all epochs; "
        "list of values (length = num_epochs) sets per-epoch sizes. "
        "Default is len(train1)+len(train2) for each epoch.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed; per-epoch seeds are derived from this.",
    )
    parser.add_argument(
        "--mutually_exclusive",
        action="store_true",
        help="Ensure samples are mutually exclusive across epochs (no sample reused).",
    )
    return parser.parse_args()


def load_split(folder: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ds_dir = folder / "dataset"
    train_path = ds_dir / "train.parquet"
    val_path = ds_dir / "val.parquet"
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(f"Missing parquet files under {ds_dir}")
    return pd.read_parquet(train_path), pd.read_parquet(val_path)


def format_ratio(ratio: float) -> str:
    return f"{ratio:.3f}".rstrip("0").rstrip(".")


def mix_train(
    train1: pd.DataFrame,
    train2: pd.DataFrame,
    mix_ratio: float,
    total_size: int,
    epoch_seed: int,
    used_indices1: Set[int],
    used_indices2: Set[int],
) -> Tuple[pd.DataFrame, int, int, Set[int], Set[int], int]:
    """
    Sample with preference for unused samples to minimize repeats across epochs.
    Falls back to reusing samples only when necessary.
    
    Returns:
        mixed_df, n1, n2, new_used_indices1, new_used_indices2, num_reused
    """
    n1 = int(round(total_size * mix_ratio))
    n2 = total_size - n1
    rng = np.random.default_rng(epoch_seed)
    
    def sample_minimize_repeats(
        df: pd.DataFrame, n: int, used_indices: Set[int], seed: int
    ) -> Tuple[List[int], int]:
        """Sample n indices, prioritizing unused ones. Returns (indices, num_reused)."""
        all_indices = list(range(len(df)))
        available = [i for i in all_indices if i not in used_indices]
        
        local_rng = np.random.default_rng(seed)
        
        if n <= len(available):
            # Enough unused samples - no repeats needed
            sampled = local_rng.choice(available, size=n, replace=False).tolist()
            return sampled, 0
        else:
            # Use all available unused samples first
            sampled = list(available)
            needed = n - len(available)
            # Fill remainder from already-used samples
            already_used = [i for i in all_indices if i in used_indices]
            if needed <= len(already_used):
                extras = local_rng.choice(already_used, size=needed, replace=False).tolist()
            else:
                # Need to sample with replacement from used samples
                extras = local_rng.choice(already_used, size=needed, replace=True).tolist()
            sampled.extend(extras)
            local_rng.shuffle(sampled)
            return sampled, needed
    
    sampled_idx1, reused1 = sample_minimize_repeats(train1, n1, used_indices1, epoch_seed)
    sampled_idx2, reused2 = sample_minimize_repeats(train2, n2, used_indices2, epoch_seed + 1)
    
    part1 = train1.iloc[sampled_idx1].copy()
    part2 = train2.iloc[sampled_idx2].copy()
    
    mixed = pd.concat([part1, part2], ignore_index=True)
    mixed = mixed.sample(frac=1.0, random_state=epoch_seed + 2).reset_index(drop=True)
    
    new_used1 = used_indices1 | set(sampled_idx1)
    new_used2 = used_indices2 | set(sampled_idx2)
    
    return mixed, n1, n2, new_used1, new_used2, reused1 + reused2


def mix_train_exclusive(
    train1: pd.DataFrame,
    train2: pd.DataFrame,
    mix_ratio: float,
    total_size: int,
    epoch_seed: int,
    used_indices1: Set[int],
    used_indices2: Set[int],
) -> Tuple[pd.DataFrame, int, int, Set[int], Set[int]]:
    """Sample mutually exclusive rows across epochs (no sample reused)."""
    available1 = [i for i in range(len(train1)) if i not in used_indices1]
    available2 = [i for i in range(len(train2)) if i not in used_indices2]

    n1 = int(round(total_size * mix_ratio))
    n2 = total_size - n1

    if n1 > len(available1):
        raise ValueError(
            f"Not enough unused samples in dataset1: need {n1}, have {len(available1)} available. "
            f"Total: {len(train1)}, already used: {len(used_indices1)}."
        )
    if n2 > len(available2):
        raise ValueError(
            f"Not enough unused samples in dataset2: need {n2}, have {len(available2)} available. "
            f"Total: {len(train2)}, already used: {len(used_indices2)}."
        )

    rng = np.random.default_rng(epoch_seed)
    sampled_idx1 = rng.choice(available1, size=n1, replace=False).tolist()
    sampled_idx2 = rng.choice(available2, size=n2, replace=False).tolist()

    part1 = train1.iloc[sampled_idx1].copy()
    part2 = train2.iloc[sampled_idx2].copy()

    mixed = pd.concat([part1, part2], ignore_index=True)
    mixed = mixed.sample(frac=1.0, random_state=epoch_seed + 2).reset_index(drop=True)

    new_used1 = used_indices1 | set(sampled_idx1)
    new_used2 = used_indices2 | set(sampled_idx2)

    return mixed, n1, n2, new_used1, new_used2


def save_run_manifest(root: Path, base_args: Dict, epochs_meta: list) -> None:
    manifest = {
        "base_args": base_args,
        "epochs": epochs_meta,
    }
    with open(root / "mix_run_args.json", "w") as f:
        json.dump(manifest, f, indent=2)


def main() -> None:
    args = parse_args()
    if len(args.mix_ratios) != args.num_epochs:
        raise ValueError("mix_ratios length must equal num_epochs (one ratio per epoch).")
    for r in args.mix_ratios:
        if not 0 < r < 1:
            raise ValueError("Each mix ratio must be between 0 and 1 (exclusive).")

    folder1 = Path(args.dataset_folder1)
    folder2 = Path(args.dataset_folder2)
    train1, val1 = load_split(folder1)
    train2, val2 = load_split(folder2)

    default_size = len(train1) + len(train2)
    if args.total_train_size is None:
        epoch_sizes = [default_size] * args.num_epochs
    elif len(args.total_train_size) == 1:
        epoch_sizes = [args.total_train_size[0]] * args.num_epochs
    elif len(args.total_train_size) == args.num_epochs:
        epoch_sizes = args.total_train_size
    else:
        raise ValueError(
            f"total_train_size must be a single value or have length = num_epochs ({args.num_epochs}), "
            f"got {len(args.total_train_size)} values."
        )

    combined_val = pd.concat([val1, val2], ignore_index=True)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    epochs_meta: List[Dict] = []
    used_indices1: Set[int] = set()
    used_indices2: Set[int] = set()

    for epoch in range(1, args.num_epochs + 1):
        mix_ratio = args.mix_ratios[epoch - 1]
        total_size = epoch_sizes[epoch - 1]
        ratio_str = format_ratio(mix_ratio)
        epoch_seed = args.seed + epoch * 997  # spaced seeds per epoch

        if args.mutually_exclusive:
            mixed_train_df, n1, n2, used_indices1, used_indices2 = mix_train_exclusive(
                train1=train1,
                train2=train2,
                mix_ratio=mix_ratio,
                total_size=total_size,
                epoch_seed=epoch_seed,
                used_indices1=used_indices1,
                used_indices2=used_indices2,
            )
            num_reused = 0
        else:
            mixed_train_df, n1, n2, used_indices1, used_indices2, num_reused = mix_train(
                train1=train1,
                train2=train2,
                mix_ratio=mix_ratio,
                total_size=total_size,
                epoch_seed=epoch_seed,
                used_indices1=used_indices1,
                used_indices2=used_indices2,
            )
            if num_reused > 0:
                print(f"  ⚠️  Epoch {epoch}: {num_reused} samples reused (not enough unique samples)")

        epoch_dir = output_root / f"split2b_mix_{ratio_str}_epoch_{epoch}"
        dataset_dir = epoch_dir / "dataset"
        dataset_dir.mkdir(parents=True, exist_ok=True)

        mixed_train_df.to_parquet(dataset_dir / "train.parquet", index=False)
        combined_val.to_parquet(dataset_dir / "val.parquet", index=False)

        epoch_args = {
            "epoch": epoch,
            "epoch_seed": epoch_seed,
            "mix_ratio": mix_ratio,
            "total_train_size": total_size,
            "from_folder1": n1,
            "from_folder2": n2,
            "samples_reused": num_reused,
            "unique_samples_used": {"folder1": len(used_indices1), "folder2": len(used_indices2)},
            "source_train_rows": {"folder1": len(train1), "folder2": len(train2)},
            "source_val_rows": {"folder1": len(val1), "folder2": len(val2)},
            "dataset_folder1": str(folder1),
            "dataset_folder2": str(folder2),
            "output_dir": str(epoch_dir),
            "mutually_exclusive": args.mutually_exclusive,
        }

        with open(epoch_dir / "mix_args.json", "w") as f:
            json.dump(epoch_args, f, indent=2)

        epochs_meta.append(epoch_args)

    base_args = {
        "dataset_folder1": str(folder1),
        "dataset_folder2": str(folder2),
        "mix_ratios": args.mix_ratios,
        "num_epochs": args.num_epochs,
        "output_root": str(output_root),
        "total_train_sizes": epoch_sizes,
        "seed": args.seed,
        "mutually_exclusive": args.mutually_exclusive,
    }
    save_run_manifest(output_root, base_args, epochs_meta)
    print(f"✅ Generated {args.num_epochs} mixed epoch(s) under {output_root}")


if __name__ == "__main__":
    main()