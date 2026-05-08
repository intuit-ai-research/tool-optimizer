#!/usr/bin/env python3
import argparse
import os

import pyarrow as pa
import pyarrow.parquet as pq


def concat_split(split, input_dirs, output_dir):
    tables = []
    for input_dir in input_dirs:
        file_path = os.path.join(input_dir, f"{split}.parquet")
        tables.append(pq.read_table(file_path))
    combined = pa.concat_tables(tables, promote=True)
    out_path = os.path.join(output_dir, f"{split}.parquet")
    pq.write_table(combined, out_path)
    return combined.num_rows, combined.schema


def dataset_name_from_path(path):
    norm_path = os.path.normpath(path)
    base = os.path.basename(norm_path)
    if base == "dataset":
        return os.path.basename(os.path.dirname(norm_path))
    return base


def default_output_dir(dataset1, dataset2):
    dataset1_parent = os.path.dirname(os.path.normpath(dataset1))
    dataset2_parent = os.path.dirname(os.path.normpath(dataset2))
    common_parent = os.path.commonpath([dataset1_parent, dataset2_parent])
    name1 = dataset_name_from_path(dataset1)
    name2 = dataset_name_from_path(dataset2)
    return os.path.join(common_parent, f"{name1}_{name2}", "dataset")


def main():
    parser = argparse.ArgumentParser(
        description="Concatenate train/val parquet from two dataset folders."
    )
    parser.add_argument(
        "--dataset1",
        required=True,
        help="Path to first dataset folder (the one containing train/val parquet).",
    )
    parser.add_argument(
        "--dataset2",
        required=True,
        help="Path to second dataset folder (the one containing train/val parquet).",
    )
    parser.add_argument(
        "--output",
        required=False,
        default=None,
        help=(
            "Path to output dataset folder (train/val parquet will be written here). "
            "Defaults to <common_parent>/<dataset1_name>_<dataset2_name>/dataset."
        ),
    )
    args = parser.parse_args()

    output_dir = args.output or default_output_dir(args.dataset1, args.dataset2)
    os.makedirs(output_dir, exist_ok=True)
    for split in ["train", "val"]:
        rows, schema = concat_split(split, [args.dataset1, args.dataset2], output_dir)
        print(f"{split}: {rows} rows")
        print(schema)
    print(f"Wrote: {output_dir}")


if __name__ == "__main__":
    main()
