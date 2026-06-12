from argparse import ArgumentParser
from pathlib import Path
from object_centric_bench.datum import CLEVRER_Video

DATASET_MAP = {
    "clevrer": CLEVRER_Video,
}


def main():
    parser = ArgumentParser(
        description="Convert raw datasets to LMDB format for training."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="clevrer",
        choices=list(DATASET_MAP.keys()),
        help="Dataset name (maps to a converter class).",
    )
    parser.add_argument(
        "--src_dir",
        type=str,
        default="./datasets/CLEVRER",
        help="Path to the raw dataset directory.",
    )
    parser.add_argument(
        "--dst_dir",
        type=str,
        default="./datasets/clevrer",
        help="Output directory for the converted LMDB files.",
    )
    args = parser.parse_args()

    converter_cls = DATASET_MAP[args.dataset]
    converter_cls.convert_dataset(
        src_dir=Path(args.src_dir),
        dst_dir=Path(args.dst_dir),
    )


if __name__ == "__main__":
    main()