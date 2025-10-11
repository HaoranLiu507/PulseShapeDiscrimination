import argparse
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class ComboConfig:
    amplitude: Tuple[float, float]
    shift: Tuple[int, int]


@dataclass(frozen=True)
class DatasetInfo:
    split: str
    path: str
    class_name: str
    stem: str
    variant: str
    variant_key: str


def load_txt_matrix(path: str) -> Tuple[np.ndarray, Optional[str]]:
    header: Optional[str] = None
    with open(path, "r", encoding="utf-8") as handle:
        first = handle.readline()
        if first.lstrip().startswith("#"):
            header = first.rstrip("\n")
    data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        data = data[np.newaxis, :]
    return data.astype(np.float64, copy=False), header


def save_txt_matrix(path: str, data: np.ndarray, header: Optional[str]) -> None:
    if header:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(header)
            if not header.endswith("\n"):
                handle.write("\n")
        with open(path, "ab") as handle:
            np.savetxt(handle, data, fmt="%1.6f")
    else:
        np.savetxt(path, data, fmt="%1.6f")


def validate_range(values: Tuple[float, float], label: str) -> Tuple[float, float]:
    low, high = values
    if low > high:
        raise ValueError(f"Invalid range for {label}: {low} > {high}")
    return float(low), float(high)


def validate_shift(values: Tuple[int, int], label: str) -> Tuple[int, int]:
    low, high = values
    if low < 0 or high < 0:
        raise ValueError(f"Negative shift in {label}")
    if low > high:
        raise ValueError(f"Invalid shift for {label}: {low} > {high}")
    return int(low), int(high)


def infer_class_name(stem: str) -> str:
    tokens = stem.lower().split("_")
    skip = {"train", "test", "validation", "validate", "val", "noise"}
    for token in reversed(tokens):
        if not token:
            continue
        if token in skip:
            continue
        if token.startswith("snr"):
            continue
        if token.startswith("class"):
            return token
        if token.isalpha():
            return token
    return stem.lower()


def extract_variant(stem: str, class_name: str) -> Tuple[str, str]:
    tokens = stem.split("_")
    variant_tokens: List[str] = []
    removed = False
    skip_tokens = {"train", "test", "validation", "validate", "val"}
    for token in tokens:
        lower = token.lower()
        if not removed and lower == class_name:
            removed = True
            continue
        if lower in skip_tokens:
            continue
        variant_tokens.append(token)
    if not variant_tokens:
        variant_tokens = [stem]
    variant = "_".join(variant_tokens)
    variant_key = "_".join(token.lower() for token in variant_tokens)
    return variant, variant_key


def find_datasets(data_root: str, include_splits: Sequence[str]) -> List[DatasetInfo]:
    discovered: List[DatasetInfo] = []
    for split in include_splits:
        split_dir = os.path.join(data_root, split)
        if not os.path.isdir(split_dir):
            continue
        for name in os.listdir(split_dir):
            if not name.lower().endswith(".txt"):
                continue
            stem, _ = os.path.splitext(name)
            class_name = infer_class_name(stem)
            variant, variant_key = extract_variant(stem, class_name)
            discovered.append(
                DatasetInfo(
                    split=split,
                    path=os.path.join(split_dir, name),
                    class_name=class_name,
                    stem=stem,
                    variant=variant,
                    variant_key=variant_key,
                )
            )
    return discovered


def build_output_path(input_path: str, suffix: str) -> str:
    directory, name = os.path.split(input_path)
    stem, ext = os.path.splitext(name)
    return os.path.join(directory, f"{stem}{suffix}{ext}")


def prompt_for_factor() -> float:
    while True:
        value = input("Enter pile-up factor (non-negative number): ").strip()
        try:
            factor = float(value)
        except ValueError:
            print("Invalid number. Please enter a numeric factor.")
            continue
        if factor < 0:
            print("Factor must be non-negative.")
            continue
        return factor


def pad_signal(signal: np.ndarray, length: int) -> np.ndarray:
    if signal.shape[0] == length:
        return signal
    if signal.shape[0] > length:
        return signal[:length]
    result = np.zeros(length, dtype=np.float64)
    result[: signal.shape[0]] = signal
    return result


def create_pileup(
    first: np.ndarray,
    second: np.ndarray,
    shift: int,
    amplitude: float,
    length: int,
) -> np.ndarray:
    base = pad_signal(first, length)
    result = base.copy()
    if shift < 0:
        raise ValueError("Shift must be non-negative")
    if shift >= length:
        return result
    scaled = second * amplitude
    available = length - shift
    if available <= 0:
        return result
    segment = scaled[:available]
    result[shift : shift + segment.shape[0]] += segment
    return result


def sample_index(size: int, rng: np.random.Generator) -> int:
    return int(rng.integers(0, size))


def distribute_counts(total: int, keys: Sequence[int]) -> Dict[int, int]:
    if not keys:
        return {}
    base = total // len(keys)
    remainder = total % len(keys)
    counts: Dict[int, int] = {key: base for key in keys}
    for key in keys[:remainder]:
        counts[key] += 1
    return counts


def generate_pair_pileups(
    first_set: np.ndarray,
    second_set: np.ndarray,
    label_pair: Tuple[str, str],
    cfg: ComboConfig,
    count: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, List[Dict[str, object]]]:
    if count <= 0:
        return np.empty((0, first_set.shape[1]), dtype=np.float64), []
    output = np.zeros((count, first_set.shape[1]), dtype=np.float64)
    metadata: List[Dict[str, object]] = []
    amp_low, amp_high = cfg.amplitude
    shift_low, shift_high = cfg.shift
    for row in range(count):
        first_idx = sample_index(first_set.shape[0], rng)
        second_idx = sample_index(second_set.shape[0], rng)
        amplitude = float(rng.uniform(amp_low, amp_high))
        shift = int(rng.integers(shift_low, shift_high + 1)) if shift_high >= shift_low else shift_low
        pile = create_pileup(
            first_set[first_idx, :],
            second_set[second_idx, :],
            shift,
            amplitude,
            first_set.shape[1],
        )
        output[row, :] = pile
        metadata.append(
            {
                "pair": f"{label_pair[0]}_{label_pair[1]}",
                "first_index": int(first_idx),
                "second_index": int(second_idx),
                "amplitude": amplitude,
                "shift": int(shift),
            }
        )
    return output, metadata


def coalesce_variant_pairs(datasets: List[DatasetInfo]) -> Dict[str, Dict[str, DatasetInfo]]:
    grouped: Dict[str, Dict[str, DatasetInfo]] = defaultdict(dict)
    for info in datasets:
        grouped[info.variant_key][info.class_name] = info
    return grouped


def generate_variant_pileups(
    variant_key: str,
    class_map: Dict[str, DatasetInfo],
    factor: float,
    rng: np.random.Generator,
    combo_defaults: ComboConfig,
    overrides: Dict[Tuple[str, str], ComboConfig],
    total_count: Optional[int] = None,
) -> Dict[Tuple[str, str], Tuple[np.ndarray, Optional[str], List[Dict[str, object]]]]:
    class_names = sorted(class_map.keys())
    if len(class_names) < 2:
        raise ValueError(f"Variant '{variant_key}' does not contain at least two classes")

    class_data: Dict[str, np.ndarray] = {}
    headers: Dict[str, Optional[str]] = {}
    signal_length: Optional[int] = None
    lengths: List[int] = []
    for class_name in class_names:
        info = class_map[class_name]
        data, header = load_txt_matrix(info.path)
        if signal_length is None:
            signal_length = data.shape[1]
        if data.shape[1] != signal_length:
            raise ValueError(
                f"Mismatched waveform lengths within variant '{variant_key}' (class {class_name})"
            )
        class_data[class_name] = data
        headers[class_name] = header
        lengths.append(int(data.shape[0]))

    if signal_length is None:
        return {}

    combos: List[Tuple[str, str]] = []
    for class_a in class_names:
        for class_b in class_names:
            combos.append((class_a, class_b))

    outputs: Dict[Tuple[str, str], Tuple[np.ndarray, Optional[str], List[Dict[str, object]]]] = {}

    for class_a, class_b in combos:
        labels = (class_a, class_b)
        cfg = overrides.get(labels, combo_defaults)
        data_a = class_data[class_a]
        data_b = class_data[class_b]
        reference = (data_a.shape[0] + data_b.shape[0]) / 2.0
        if total_count is not None:
            count = max(total_count, 0)
        else:
            count = int(round(reference * factor))
            count = max(count, 0)
        prepared, metadata = generate_pair_pileups(
            data_a,
            data_b,
            labels,
            cfg,
            count,
            rng,
        )
        outputs[labels] = (prepared, headers[class_a], metadata)

    return outputs


def write_metadata(path: str, rows: List[Dict[str, object]]) -> None:
    import csv

    if not rows:
        with open(path, "w", newline="", encoding="utf-8") as handle:
            handle.write("")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate pile-up datasets for all discovered class pairs.")
    parser.add_argument("--data-root", default=os.path.join("Data"), help="Root folder containing dataset splits")
    parser.add_argument(
        "--splits",
        nargs="*",
        default=["Train", "Test", "Validation"],
        help="List of splits to process (default: Train Test Validation)",
    )
    parser.add_argument("--factor", type=float, help="Multiplier applied to base sample counts")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save-metadata", action="store_true", help="Write CSV metadata alongside outputs")
    parser.add_argument(
        "--amplitude",
        type=float,
        nargs=2,
        default=(0.5, 1.5),
        metavar=("MIN", "MAX"),
        help="Amplitude range for any class pair",
    )
    parser.add_argument(
        "--shift",
        type=int,
        nargs=2,
        default=(10, 100),
        metavar=("MIN", "MAX"),
        help="Sample shift range for any class pair",
    )
    return parser.parse_args(argv or sys.argv[1:])


def ensure_factor(factor: Optional[float]) -> float:
    if factor is None:
        return prompt_for_factor()
    if factor < 0:
        raise ValueError("Factor must be non-negative")
    return factor


def compute_validation_requirement(validation_files: List[DatasetInfo], factor: float) -> int:
    if not validation_files:
        return 0
    total_rows = 0
    for info in validation_files:
        data, _ = load_txt_matrix(info.path)
        total_rows += data.shape[0]
    return int(round(total_rows * factor))


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    data_root = args.data_root
    if not os.path.isdir(data_root):
        print(f"Data root not found: {data_root}")
        sys.exit(1)

    factor = ensure_factor(args.factor)
    amplitude_range = validate_range(tuple(args.amplitude), "amplitude")
    shift_range = validate_shift(tuple(args.shift), "shift")
    combo_defaults = ComboConfig(amplitude_range, shift_range)
    overrides: Dict[Tuple[str, str], ComboConfig] = {}

    splits = [split.strip() for split in args.splits if split.strip()]
    datasets = find_datasets(data_root, splits)
    if not datasets:
        print("No dataset files discovered.")
        sys.exit(1)

    rng = np.random.default_rng(args.seed)

    grouped_by_split: Dict[str, List[DatasetInfo]] = defaultdict(list)
    for info in datasets:
        grouped_by_split[info.split].append(info)

    validation_files = [info for info in datasets if info.split.lower().startswith("validation")]
    required_validation = compute_validation_requirement(validation_files, factor)

    metadata_records: Dict[str, List[Dict[str, object]]] = defaultdict(list)

    for split, split_datasets in grouped_by_split.items():
        if split.lower().startswith("validation"):
            continue
        variants = coalesce_variant_pairs(split_datasets)
        for variant_key, class_map in variants.items():
            try:
                outputs = generate_variant_pileups(
                    variant_key,
                    class_map,
                    factor,
                    rng,
                    combo_defaults,
                    overrides,
                )
            except ValueError as exc:
                print(f"Skipping variant '{variant_key}' in split '{split}': {exc}")
                continue
            if not outputs:
                continue
            variant_dir = os.path.dirname(next(iter(class_map.values())).path)
            variant_name = next(iter(class_map.values())).variant
            for (class_a, class_b), (array, header, metadata) in outputs.items():
                if array.size == 0:
                    continue
                base_variant = variant_name or f"{class_a}_{class_b}"
                if class_a == class_b:
                    output_name = f"{class_map[class_a].stem}_pileup.txt"
                else:
                    output_name = f"{base_variant}_pileup_{class_a}_{class_b}.txt"
                output_path = os.path.join(variant_dir, output_name)
                save_txt_matrix(output_path, array, header)
                print(f"Saved {split} pile-up data: {output_path}")
                if args.save_metadata:
                    metadata_records[output_path].extend(metadata)

    if required_validation > 0:
        test_datasets = grouped_by_split.get("Test", [])
        if not test_datasets:
            print("No test datasets available to build validation pile-ups.")
            sys.exit(1)
        variants = coalesce_variant_pairs(test_datasets)
        if not variants:
            print("No test variants available to build validation pile-ups.")
            sys.exit(1)
        validation_dir = os.path.join(data_root, "Validation")
        os.makedirs(validation_dir, exist_ok=True)
        for variant_key, class_map in variants.items():
            try:
                outputs = generate_variant_pileups(
                    variant_key,
                    class_map,
                    factor,
                    rng,
                    combo_defaults,
                    overrides,
                )
            except ValueError as exc:
                print(f"Skipping validation variant '{variant_key}': {exc}")
                continue
            if not outputs:
                continue
            variant_name = next(iter(class_map.values())).variant
            combined_rows: List[np.ndarray] = []
            combined_meta: List[Dict[str, object]] = []
            header: Optional[str] = None
            for (class_a, class_b), (array, array_header, metadata) in outputs.items():
                if array.size == 0:
                    continue
                if header is None:
                    header = array_header
                combined_rows.append(array)
                combined_meta.extend(metadata)
            if not combined_rows:
                continue
            combined = np.vstack(combined_rows)
            output_name = f"{variant_name}_pileup.txt"
            validation_path = os.path.join(validation_dir, output_name)
            save_txt_matrix(validation_path, combined, header)
            print(f"Saved validation pile-up data: {validation_path}")
            if args.save_metadata:
                metadata_records[validation_path].extend(combined_meta)

    if args.save_metadata:
        for output_path, records in metadata_records.items():
            meta_path = build_output_path(output_path, "_metadata")
            write_metadata(meta_path, records)
            print(f"Saved metadata: {meta_path}")


if __name__ == "__main__":
    main()

