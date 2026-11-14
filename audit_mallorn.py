"""Audit the original MALLORN competition files before building features."""

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path


def audit_partition(root, partition):
    with (root / f"{partition}_log.csv").open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    objects = {row["object_id"]: row for row in rows}
    if not rows or len(objects) != len(rows):
        raise ValueError(f"{partition}: expected nonempty metadata with unique IDs")
    if ("target" in rows[0]) != (partition == "train"):
        raise ValueError(f"{partition}: unexpected target column for original release")

    shards = defaultdict(set)
    for row in rows:
        shards[row["split"]].add(row["object_id"])

    bands = Counter()
    coverage = defaultdict(set)
    negative_flux = 0
    invalid_flux = 0
    invalid_errors = 0
    for shard, expected in sorted(shards.items()):
        path = root / shard / f"{partition}_full_lightcurves.csv"
        seen = set()
        with path.open(encoding="utf-8-sig", newline="") as file:
            for row in csv.DictReader(file):
                object_id = row["object_id"]
                band = row["Filter"]
                seen.add(object_id)
                bands[band] += 1
                coverage[band].add(object_id)
                flux = parse_number(row["Flux"])
                invalid_flux += not math.isfinite(flux)
                negative_flux += math.isfinite(flux) and flux < 0
                invalid_errors += not (0 < parse_number(row["Flux_err"]) < math.inf)
        if seen != expected:
            raise ValueError(
                f"{path}: {len(expected - seen)} IDs missing, {len(seen - expected)} unexpected"
            )

    print(f"{partition}: {len(objects)} objects, {sum(bands.values())} observations")
    print(f"  Columns: {', '.join(rows[0])}")
    for field in ("Z", "Z_err", "EBV"):
        missing = sum(row.get(field, "").strip().lower() in {"", "nan", "na"} for row in rows)
        print(f"  Missing {field}: {missing}")
    if partition == "train":
        counts = Counter(row["target"] for row in rows)
        if set(counts) != {"0", "1"}:
            raise ValueError("Training target must contain both binary classes")
        print(f"  Target counts: {dict(sorted(counts.items()))}")
    print(f"  Negative flux rows: {negative_flux}, invalid flux errors: {invalid_errors}")
    print(f"  Missing or invalid flux rows: {invalid_flux}")
    print("  Band: observations, objects without coverage")
    for band in sorted(set("ugrizy") | bands.keys()):
        print(f"    {band}: {bands[band]}, {len(objects) - len(coverage[band])}")
    return objects.keys()


def parse_number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path, help="Directory containing the two logs and split folders")
    args = parser.parse_args()
    train = audit_partition(args.data_dir, "train")
    test = audit_partition(args.data_dir, "test")
    if train & test:
        raise ValueError("Training and test metadata share object IDs")


if __name__ == "__main__":
    main()
