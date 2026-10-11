import json
from pathlib import Path
from collections import Counter

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.warp import reproject
from shapely.geometry import mapping
from shapely.wkt import loads as wkt_loads

from src.constants import XBD_S12_PATH
from src.data.metadata import load_metadata


ORIGINAL_XBD = Path(
    r"D:\ShoreGate\xbd_original\xview2_geotiff\geotiffs"
)
THRESHOLD = 0.50
SIZE = 128

DAMAGE_DICT = {
    "no-damage": 1,
    "minor-damage": 2,
    "major-damage": 3,
    "destroyed": 4,
    "un-classified": 5,
}



def main():
    metadata = load_metadata()
    palu = metadata[metadata["disaster"] == "palu-tsunami"]

    total = Counter()
    patch_rows = []
    skipped = []

    for uid, row in palu.iterrows():
        tier = "tier1" if row["xbd_tier"] == "train" else row["xbd_tier"]

        json_path = (
            ORIGINAL_XBD / tier / "labels"
            / f"{uid}_post_disaster.json"
        )
        s2_path = XBD_S12_PATH / "s2" / f"{uid}_post_disaster_s2.tif"

        if not json_path.exists() or not s2_path.exists():
            skipped.append((uid, "missing JSON or S2"))
            continue

        with rasterio.open(s2_path) as ds:
            dst_transform = ds.transform
            dst_crs = ds.crs
            height, width = ds.height, ds.width
            if (height, width) != (SIZE, SIZE):
                skipped.append((uid, "unexpected S2 dimensions"))
                continue

        with rasterio.open(
            ORIGINAL_XBD / tier / "images" / f"{uid}_post_disaster.tif"
        ) as src:
            src_transform = src.transform
            src_crs = src.crs
            src_shape = (src.height, src.width)

        data = json.loads(json_path.read_text(encoding="utf-8"))
        labels = np.zeros(src_shape, dtype=np.uint8)

        for feature in data["features"]["xy"]:
            subtype = feature["properties"]["subtype"]
            value = DAMAGE_DICT[subtype]
            polygon = wkt_loads(feature["wkt"])

            polygon_mask = rasterize(
                [(mapping(polygon), value)],
                out_shape=src_shape,
                transform=rasterio.Affine.identity(),
                fill=0,
                dtype="uint8",
            )
            labels[polygon_mask > 0] = value

        coverage = {}
        for name, values in {
            "intact": [1],
            "damaged": [2, 3, 4],
            "unclassified": [5],
        }.items():
            source_mask = np.isin(labels, values).astype(np.float32)
            target_mask = np.zeros((height, width), dtype=np.float32)

            reproject(
                source=source_mask,
                destination=target_mask,
                src_transform=src_transform,
                src_crs=src_crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.average,
            )
            coverage[name] = target_mask

        # Same initial 50% threshold as the sample preparation script.
        y = np.full((height, width), 255, dtype=np.uint8)
        y[
            (coverage["intact"] >= THRESHOLD)
            & (coverage["unclassified"] < THRESHOLD)
        ] = 0
        y[
            (coverage["damaged"] >= THRESHOLD)
            & (coverage["unclassified"] < THRESHOLD)
        ] = 1

        intact_n = int(np.count_nonzero(y == 0))
        damaged_n = int(np.count_nonzero(y == 1))
        excluded_n = int(np.count_nonzero(y == 255))

        total.update({
            "intact": intact_n,
            "damaged": damaged_n,
            "excluded": excluded_n,
        })

        patch_rows.append({
            "uid": uid,
            "tier": row["xbd_tier"],
            "intact": intact_n,
            "damaged": damaged_n,
            "excluded": excluded_n,
            "labeled_total": intact_n + damaged_n,
            "damaged_pct": (
                100 * damaged_n / (intact_n + damaged_n)
                if intact_n + damaged_n else 0
            ),
        })

        print(
            f"{uid}: intact={intact_n}, damaged={damaged_n}, "
            f"excluded={excluded_n}"
        )

    print("\n=== PALU SUMMARY ===")
    print("Metadata patches:", len(palu))
    print("Processed patches:", len(patch_rows))
    print("Skipped patches:", len(skipped))
    print("Total intact:", total["intact"])
    print("Total damaged:", total["damaged"])
    print("Total labeled:", total["intact"] + total["damaged"])
    print("Total excluded:", total["excluded"])

    labeled = total["intact"] + total["damaged"]
    if labeled:
        print(f"Damaged percentage: {100 * total['damaged'] / labeled:.3f}%")
        print(f"Intact percentage: {100 * total['intact'] / labeled:.3f}%")
        print(f"Intact:damaged ratio: {total['intact'] / max(total['damaged'], 1):.2f}:1")

    if patch_rows:
        out_dir = XBD_S12_PATH / "processed"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_csv = out_dir / "palu_label_summary.csv"

        import csv
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=patch_rows[0].keys())
            writer.writeheader()
            writer.writerows(patch_rows)

        print("Per-patch summary saved:", out_csv)

    if skipped:
        print("\nSkipped samples (first 20):")
        for item in skipped[:20]:
            print(" ", item)


if __name__ == "__main__":
    main()
