from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from rasterio.features import rasterize
from rasterio.warp import reproject, Resampling
from shapely import wkt

from src.constants import PROJECT_PATH, XBD_S12_PATH
from src.data.metadata import load_metadata


# ============================================================
# Configuration
# ============================================================

ORIGINAL_XBD = PROJECT_PATH.parent / "xbd_original" / "xview2_geotiff" / "geotiffs"

OUTPUT_DIR = XBD_S12_PATH / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_DATASET = OUTPUT_DIR / "palu_rf_dataset.npz"
OUTPUT_SUMMARY = OUTPUT_DIR / "palu_rf_patch_summary.csv"

COVERAGE_THRESHOLD = 0.50
S2_BAND_NUMBERS = [2, 3, 4, 8, 11, 12]

# xBD damage labels:
# 1 = no-damage, 2 = minor, 3 = major,
# 4 = destroyed, 5 = un-classified
INTACT_CLASSES = {1}
DAMAGED_CLASSES = {2, 3, 4}
UNCLASSIFIED_CLASSES = {5}

FEATURE_NAMES = [
    "S1_pre_VV",
    "S1_pre_VH",
    "S1_post_VV",
    "S1_post_VH",
    "S2_pre_B2",
    "S2_pre_B3",
    "S2_pre_B4",
    "S2_pre_B8",
    "S2_pre_B11",
    "S2_pre_B12",
    "S2_post_B2",
    "S2_post_B3",
    "S2_post_B4",
    "S2_post_B8",
    "S2_post_B11",
    "S2_post_B12",
    "S1_change_VV",
    "S1_change_VH",
    "S2_change_B2",
    "S2_change_B3",
    "S2_change_B4",
    "S2_change_B8",
    "S2_change_B11",
    "S2_change_B12",
]

assert len(FEATURE_NAMES) == 24


# ============================================================
# Helpers
# ============================================================

def get_tier(row):
    """Map metadata xbd_tier to the original xBD folder."""
    tier = str(row["xbd_tier"]).lower()
    return "tier1" if tier == "train" else tier


def load_feature_image(path, indexes=None):
    """Read selected bands as float32; convert NoData to NaN."""
    with rasterio.open(path) as src:
        if indexes is None:
            indexes = list(range(1, src.count + 1))

        if max(indexes) > src.count:
            raise ValueError(
                f"Requested band {max(indexes)}, but {path.name} "
                f"has only {src.count} bands."
            )

        arr = src.read(indexes=indexes, masked=True)
        arr = arr.astype(np.float32).filled(np.nan)

        profile = {
            "height": src.height,
            "width": src.width,
            "transform": src.transform,
            "crs": src.crs,
        }

    return arr, profile


def make_labels(json_path, target_profile, source_profile, supersample=8):
    """
    Rasterize xBD xy-coordinate polygons onto the Sentinel-2 grid.

    Labels:
        0   = intact
        1   = damaged
        255 = excluded / ambiguous
    """
    from pyproj import Transformer
    from shapely import transform as shapely_transform

    if supersample < 1:
        raise ValueError("supersample must be >= 1")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    features_data = data.get("features", {})
    if not isinstance(features_data, dict) or "xy" not in features_data:
        raise ValueError("Expected xBD JSON features['xy'] group.")

    # Use only the pixel-coordinate representation.
    features = features_data["xy"]

    class_map = {
        "no-damage": 1,
        "minor-damage": 2,
        "major-damage": 3,
        "destroyed": 4,
        "un-classified": 5,
    }

    source_transform = source_profile["transform"]
    source_crs = source_profile["crs"]
    target_crs = target_profile["crs"]

    source_to_target = Transformer.from_crs(
        source_crs, target_crs, always_xy=True
    )

    def pixel_to_source_map(coords):
        x = (
            source_transform.a * coords[:, 0]
            + source_transform.b * coords[:, 1]
            + source_transform.c
        )
        y = (
            source_transform.d * coords[:, 0]
            + source_transform.e * coords[:, 1]
            + source_transform.f
        )
        return np.column_stack((x, y))

    def source_map_to_target(coords):
        x, y = source_to_target.transform(
            coords[:, 0], coords[:, 1]
        )
        return np.column_stack((x, y))

    shapes_by_class = {i: [] for i in class_map.values()}
    coordinate_counts = {
        "xy": 0,
        "skipped_empty": 0,
        "skipped_unknown_class": 0,
    }

    for index, feature in enumerate(features):
        properties = feature.get("properties", {})
        subtype = properties.get("subtype")
        wkt_string = feature.get("wkt")

        if subtype not in class_map or not wkt_string:
            coordinate_counts["skipped_unknown_class"] += 1
            continue

        geometry = wkt.loads(wkt_string)

        if geometry.is_empty:
            coordinate_counts["skipped_empty"] += 1
            continue

        if not geometry.is_valid:
            raise ValueError(
                f"Invalid xy polygon at feature {index}: "
                f"{geometry.bounds}"
            )

        if not np.all(np.isfinite(geometry.bounds)):
            raise ValueError(
                f"Non-finite xy coordinates at feature {index}"
            )

        map_geometry = shapely_transform(
            geometry, pixel_to_source_map
        )
        projected_geometry = shapely_transform(
            map_geometry, source_map_to_target
        )

        if (
            projected_geometry.is_empty
            or not projected_geometry.is_valid
            or not np.all(np.isfinite(projected_geometry.bounds))
        ):
            raise ValueError(
                f"Invalid projected polygon at feature {index}"
            )

        shapes_by_class[class_map[subtype]].append(
            (projected_geometry, 1)
        )
        coordinate_counts["xy"] += 1

    print("[DEBUG] Coordinate handling:", coordinate_counts)
    print(
        "[DEBUG] Polygons projected:",
        sum(len(v) for v in shapes_by_class.values()),
    )

    height = target_profile["height"]
    width = target_profile["width"]

    high_transform = target_profile["transform"] * Affine.scale(
        1 / supersample, 1 / supersample
    )
    high_shape = (height * supersample, width * supersample)

    coverage_by_class = {}

    for class_id, shapes in shapes_by_class.items():
        high_mask = np.zeros(high_shape, dtype=np.uint8)

        for start in range(0, len(shapes), 10):
            batch_mask = rasterize(
                shapes=shapes[start:start + 10],
                out_shape=high_shape,
                transform=high_transform,
                fill=0,
                dtype=np.uint8,
                all_touched=False,
            )
            np.maximum(high_mask, batch_mask, out=high_mask)

        coverage_by_class[class_id] = high_mask.reshape(
            height, supersample, width, supersample
        ).mean(axis=(1, 3), dtype=np.float32)

    intact_cov = coverage_by_class[1]
    damaged_cov = np.maximum.reduce([
        coverage_by_class[2],
        coverage_by_class[3],
        coverage_by_class[4],
    ])
    unclassified_cov = coverage_by_class[5]

    intact_candidate = intact_cov >= COVERAGE_THRESHOLD
    damaged_candidate = damaged_cov >= COVERAGE_THRESHOLD
    clear_class = unclassified_cov < COVERAGE_THRESHOLD

    exclusive_intact = (
        intact_candidate & ~damaged_candidate & clear_class
    )
    exclusive_damaged = (
        damaged_candidate & ~intact_candidate & clear_class
    )

    labels = np.full((height, width), 255, dtype=np.uint8)
    labels[exclusive_intact] = 0
    labels[exclusive_damaged] = 1

    return labels



# ============================================================
# Main
# ============================================================

def main():
    metadata = load_metadata()
    palu = metadata[metadata["disaster"] == "palu-tsunami"].copy()

    print("Metadata columns:", metadata.columns.tolist())
    print("Metadata index name:", metadata.index.name)
    print(metadata.head(2))

    if palu.empty:
        raise RuntimeError(
            "No Palu records found. Check load_metadata() and metadata."
        )

    print(f"Palu metadata patches: {len(palu)}")
    print(f"Coverage threshold: {COVERAGE_THRESHOLD:.2f}")
    print("Generating features and labels...\n")

    X_parts = []
    y_parts = []
    patch_id_parts = []
    summary_rows = []
    skipped = []

    for uid, row in palu.iterrows():
        uid = str(uid)
        tier = get_tier(row)

        original_dir = ORIGINAL_XBD / tier
        label_path = original_dir / "labels" / (
            f"{uid}_post_disaster.json"
        )
        source_image_path = original_dir / "images" / (
            f"{uid}_post_disaster.tif"
        )

        s1_dir = XBD_S12_PATH / "s1"
        s2_dir = XBD_S12_PATH / "s2"

        s1_pre_path = s1_dir / f"{uid}_pre_disaster_s1.tif"
        s1_post_path = s1_dir / f"{uid}_post_disaster_s1.tif"
        s2_pre_path = s2_dir / f"{uid}_pre_disaster_s2.tif"
        s2_post_path = s2_dir / f"{uid}_post_disaster_s2.tif"

        required = [
            label_path,
            source_image_path,
            s1_pre_path,
            s1_post_path,
            s2_pre_path,
            s2_post_path,
        ]

        missing = [str(p) for p in required if not p.exists()]
        if missing:
            skipped.append((uid, "Missing files: " + "; ".join(missing)))
            print(f"[SKIP] {uid}: missing required files")
            continue

        try:
            # Original xBD post-event image defines the source pixel grid.
            with rasterio.open(source_image_path) as src:
                source_height = src.height
                source_width = src.width
                source_profile = {
                    "height": src.height,
                    "width": src.width,
                    "transform": src.transform,
                    "crs": src.crs,
                }

            # Read the selected features.
            s1_pre, s1_pre_profile = load_feature_image(s1_pre_path)
            s1_post, s1_post_profile = load_feature_image(s1_post_path)

            s2_pre, s2_pre_profile = load_feature_image(
                s2_pre_path, S2_BAND_NUMBERS
            )
            s2_post, s2_post_profile = load_feature_image(
                s2_post_path, S2_BAND_NUMBERS
            )

            # Ensure all four satellite images share the same grid.
            profiles = [
                s1_pre_profile,
                s1_post_profile,
                s2_pre_profile,
                s2_post_profile,
            ]

            reference = profiles[0]
            for profile in profiles[1:]:
                same_grid = (
                    profile["height"] == reference["height"]
                    and profile["width"] == reference["width"]
                    and profile["transform"] == reference["transform"]
                    and profile["crs"] == reference["crs"]
                )
                if not same_grid:
                    raise ValueError(
                        "S1/S2 pre/post images do not share the same grid."
                    )

            target_profile = s2_post_profile

            # Feature order: 4 S1 + 12 S2 + 2 S1 changes + 6 S2 changes.
            features = np.concatenate(
                [
                    s1_pre,
                    s1_post,
                    s2_pre,
                    s2_post,
                    s1_post - s1_pre,
                    s2_post - s2_pre,
                ],
                axis=0,
            ).astype(np.float32)

            if features.shape[0] != len(FEATURE_NAMES):
                raise ValueError(
                    f"Expected 24 features, got {features.shape[0]}."
                )

            # Rasterize geographic xBD polygons directly onto the S2 grid.
            labels = make_labels(
                label_path,
                target_profile,
                source_profile,
            )

            # Exclude pixels where any required input feature is invalid.
            valid_features = np.all(np.isfinite(features), axis=0)
            labels[~valid_features] = 255

            # Flatten only labelled pixels. Keep patch ID for grouped splits.
            keep = labels != 255
            y_patch = labels[keep].astype(np.uint8)
            X_patch = features[:, keep].T.astype(np.float32)

            intact_count = int(np.sum(y_patch == 0))
            damaged_count = int(np.sum(y_patch == 1))
            excluded_count = int(labels.size - np.sum(keep))

            summary_rows.append({
                "xbd_uid": uid,
                "tier": tier,
                "intact": intact_count,
                "damaged": damaged_count,
                "labeled": intact_count + damaged_count,
                "excluded": excluded_count,
            })

            if len(y_patch) > 0:
                X_parts.append(X_patch)
                y_parts.append(y_patch)
                patch_id_parts.append(
                    np.full(len(y_patch), uid, dtype=f"<U{len(uid)}")
                )

            print(
                f"{uid}: intact={intact_count}, "
                f"damaged={damaged_count}, excluded={excluded_count}"
            )

        except Exception as exc:
            skipped.append((uid, repr(exc)))
            print(f"[SKIP] {uid}: {exc}")

    # Save patch-level processing summary even if some patches failed.
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTPUT_SUMMARY, index=False)

    if not X_parts:
        raise RuntimeError(
            "No labelled samples were generated. Check the paths and logs."
        )

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    patch_ids = np.concatenate(patch_id_parts, axis=0)

    # Defensive checks.
    if X.ndim != 2 or X.shape[1] != len(FEATURE_NAMES):
        raise RuntimeError(f"Unexpected X shape: {X.shape}")

    if len(X) != len(y) or len(y) != len(patch_ids):
        raise RuntimeError("X, y and patch_ids have inconsistent lengths.")

    if len(X) == 0:
        raise RuntimeError("Dataset contains no labelled samples.")

    if not np.all(np.isfinite(X)):
        raise RuntimeError("X contains NaN or infinite feature values.")

    if not np.all(np.isin(y, [0, 1])):
        raise RuntimeError("Unexpected label values in y.")

    if len(np.unique(y)) < 2:
        raise RuntimeError("Dataset contains only one class.")

    if np.any(pd.isna(patch_ids)):
        raise RuntimeError("patch_ids contains missing values.")

    np.savez_compressed(
        OUTPUT_DATASET,
        X=X,
        y=y,
        patch_ids=patch_ids,
        feature_names=np.asarray(FEATURE_NAMES),
    )

    total_intact = int(np.sum(y == 0))
    total_damaged = int(np.sum(y == 1))

    print("\n========== DATASET SUMMARY ==========")
    print(f"Metadata patches: {len(palu)}")
    print(f"Successfully processed: {len(summary)}")
    print(f"Skipped/failed patches: {len(skipped)}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Intact: {total_intact:,}")
    print(f"Damaged: {total_damaged:,}")
    print(f"Intact:damaged ratio: "
          f"{total_intact / max(total_damaged, 1):.2f}:1")
    print(f"Dataset saved: {OUTPUT_DATASET}")
    print(f"Patch summary saved: {OUTPUT_SUMMARY}")

    if skipped:
        print("\nSkipped patches:")
        for uid, reason in skipped:
            print(f"- {uid}: {reason}")

        print(
            "\nWARNING: Some patches failed. Review the messages above "
            "before using this dataset for training."
        )


if __name__ == "__main__":
    main()
