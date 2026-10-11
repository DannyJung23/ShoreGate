import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import reproject
from rasterio.warp import transform_bounds
from rasterio.warp import transform_geom
from shapely.wkt import loads as wkt_loads
from rasterio.features import rasterize
from rasterio.transform import Affine
from shapely.geometry import mapping
from shapely import transform as shapely_transform
from rasterio.warp import transform

from src.constants import XBD_S12_PATH
from src.data.metadata import load_metadata


DAMAGE_DICT = {
    "no-damage": 1,
    "minor-damage": 2,
    "major-damage": 3,
    "destroyed": 4,
    "un-classified": 5,
}


UID_DEFAULT = "palu-tsunami_00000000"
SOURCE_SIZE = 1024
TARGET_SIZE = 128
CLASS_GROUPS = {
    "intact": {1},
    "damaged": {2, 3, 4},
    "unclassified": {5},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-xbd-path", required=True, type=Path)
    parser.add_argument("--uid", default=UID_DEFAULT)
    args = parser.parse_args()

    metadata = load_metadata()
    if args.uid not in metadata.index:
        raise KeyError(f"{args.uid} is not in xBD-S12 metadata")

    row = metadata.loc[args.uid]
    # Match create_aligned_vrt.py: project the metadata footprint and use its bounds.
    projected = gpd.GeoSeries([row.geometry], crs="EPSG:4326").to_crs(row.best_utm)
    bounds = tuple(projected.total_bounds)
    xmin, ymin, xmax, ymax = bounds
    crs = rasterio.crs.CRS.from_user_input(row.best_utm)

    tier = "tier1" if row.xbd_tier == "train" else row.xbd_tier
    json_path = (
        args.original_xbd_path / tier / "labels" /
        f"{args.uid}_post_disaster.json"
    )
    if not json_path.exists():
        raise FileNotFoundError(f"Sample label JSON not found: {json_path}")

    with json_path.open(encoding="utf-8") as f:
        data = json.load(f)

    # Match create_masks.py overwrite behavior when polygons overlap.
    labels = np.zeros((SOURCE_SIZE, SOURCE_SIZE), dtype=np.uint8)
    for feature in data["features"]["xy"]:
        subtype = feature["properties"]["subtype"]
        class_value = DAMAGE_DICT[subtype]
        polygon = wkt_loads(feature["wkt"])
        polygon_mask = rasterize(
            [(mapping(polygon), 1)],
            out_shape=(SOURCE_SIZE, SOURCE_SIZE),
            transform=Affine.identity(),
            fill=0,
            dtype="uint8",
        )
        labels[polygon_mask > 0] = class_value

    original_image_path = (
    args.original_xbd_path
    / tier
    / "images"
    / f"{args.uid}_post_disaster.tif"
    )

    with rasterio.open(original_image_path) as source_ds:
        if source_ds.shape != (SOURCE_SIZE, SOURCE_SIZE):
            raise RuntimeError(
                f"Unexpected original xBD dimensions: {source_ds.shape}"
            )

        source_transform = source_ds.transform
        source_crs = source_ds.crs
        source_bounds_geographic = source_ds.bounds

    print("Original xBD CRS:", source_crs)
    print("Original xBD transform:", source_transform)
    target_path = (
    XBD_S12_PATH
    / "s2"
    / f"{args.uid}_post_disaster_s2.tif"
    )

    with rasterio.open(target_path) as target_ds:
        target_transform = target_ds.transform
        target_crs = target_ds.crs
        target_height = target_ds.height
        target_width = target_ds.width
        target_bounds = target_ds.bounds

    if (target_height, target_width) != (TARGET_SIZE, TARGET_SIZE):
        raise RuntimeError(
            f"Unexpected Sentinel dimensions: "
            f"{target_height}x{target_width}"
        )

    print(f"UID: {args.uid}")
    print(f"CRS: {crs}")
    print(f"Projected bounds: {bounds}")
    print(
        f"Source grid: {SOURCE_SIZE}x{SOURCE_SIZE}; "
        f"target Sentinel grid: {TARGET_SIZE}x{TARGET_SIZE}"
    )

    # Direct-rasterization comparison on the Sentinel-2 grid.
    direct_masks = {
        "intact": np.zeros((target_height, target_width), dtype=np.uint8),
        "damaged": np.zeros((target_height, target_width), dtype=np.uint8),
        "unclassified": np.zeros((target_height, target_width), dtype=np.uint8),
    }

    group_for_class = {
        1: "intact",
        2: "damaged",
        3: "damaged",
        4: "damaged",
        5: "unclassified",
    }

    with rasterio.open(original_image_path) as source_ds:
        affine = source_ds.transform

        def pixel_to_geographic(x, y, z=None):
            lon = affine.c + affine.a * x + affine.b * y
            lat = affine.f + affine.d * x + affine.e * y
            return (lon, lat) if z is None else (lon, lat, z)

        for feature in data["features"]["xy"]:
            subtype = feature["properties"]["subtype"]
            class_value = DAMAGE_DICT[subtype]
            group = group_for_class[class_value]
            polygon = wkt_loads(feature["wkt"])

            geographic_polygon = shapely_transform(
                polygon,
                lambda coords: np.column_stack((
                    source_ds.transform.c
                    + source_ds.transform.a * coords[:, 0]
                    + source_ds.transform.b * coords[:, 1],
                    source_ds.transform.f
                    + source_ds.transform.d * coords[:, 0]
                    + source_ds.transform.e * coords[:, 1],
                )),
            )

            projected_polygon = rasterio.warp.transform_geom(
                source_ds.crs,
                target_crs,
                mapping(geographic_polygon),
            )

            polygon_mask = rasterize(
                [(projected_polygon, 1)],
                out_shape=(target_height, target_width),
                transform=target_transform,
                fill=0,
                dtype="uint8",
                all_touched=False,
            )

            direct_masks[group] = np.maximum(
                direct_masks[group],
                polygon_mask,
            )

    warped_masks = {}
    for name, values in CLASS_GROUPS.items():
        source_mask = np.isin(labels, list(values)).astype(np.float32)
        target_mask = np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=np.float32)

        reproject(
            source=source_mask,
            destination=target_mask,
            src_transform=source_transform,
            src_crs=source_crs,
            dst_transform=target_transform,
            dst_crs=target_crs,
            resampling=Resampling.average,
        )
        warped_masks[name] = target_mask

        source_coverage = float(source_mask.mean())
        target_coverage = float(target_mask.mean())
        print(
            f"{name}: source coverage={source_coverage:.6%}; "
            f"128-grid mean coverage={target_coverage:.6%}; "
            f"target pixels with coverage>0="
            f"{int(np.count_nonzero(target_mask))}"
        )

    print("\nDirect-rasterization comparison:")
    for name in direct_masks:
        direct_count = int(np.count_nonzero(direct_masks[name]))
        average_count = int(
            np.count_nonzero(warped_masks[name] > 0)
        )
        print(
            f"  {name}: "
            f"average-resampling pixels={average_count}; "
            f"direct-rasterization pixels={direct_count}"
        )

    # These masks should be mutually exclusive after the source-label overwrite.
    total_fraction = sum(mask for mask in warped_masks.values())
    overlap_excess = float(np.maximum(total_fraction - 1.0, 0.0).max())

    xbd_bounds_in_target_crs = transform_bounds(
        source_crs,
        target_crs,
        *source_bounds_geographic,
        densify_pts=21,
    )

    bounds_difference = tuple(
        round(a - b, 3)
        for a, b in zip(xbd_bounds_in_target_crs, target_bounds)
    )

    print("Original xBD bounds in Sentinel CRS:", xbd_bounds_in_target_crs)
    print("Sentinel bounds:", tuple(target_bounds))
    print("Bounds difference (metres):", bounds_difference)
    print(
        "Maximum absolute bounds difference (m):",
        round(max(abs(v) for v in bounds_difference), 3),
    )


    # Visual diagnostic: project original xBD polygons onto Sentinel-2 RGB.
    from rasterio.warp import transform_geom
    from matplotlib.patches import Polygon as MplPolygon

    with rasterio.open(target_path) as ds:
        blue = ds.read(2).astype(np.float32)
        green = ds.read(3).astype(np.float32)
        red = ds.read(4).astype(np.float32)

        target_transform = ds.transform
        target_crs = ds.crs
        target_height = ds.height
        target_width = ds.width

    valid = (
        np.isfinite(red) & np.isfinite(green) & np.isfinite(blue)
        & (red != 65535) & (green != 65535) & (blue != 65535)
    )

    rgb = np.zeros((target_height, target_width, 3), dtype=np.float32)
    for i, band in enumerate([red, green, blue]):
        if valid.any():
            lo, hi = np.percentile(band[valid], [2, 98])
            if hi > lo:
                rgb[:, :, i] = np.clip((band - lo) / (hi - lo), 0, 1)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(rgb, origin="upper")

    intact_count = 0
    damaged_count = 0
    skipped_count = 0

    with rasterio.open(original_image_path) as source_ds:
        source_transform = source_ds.transform
        source_crs = source_ds.crs

        def xy_to_geographic(geometry):
            return shapely_transform(
                geometry,
                lambda coords: np.column_stack((
                    source_transform.c
                    + source_transform.a * coords[:, 0]
                    + source_transform.b * coords[:, 1],
                    source_transform.f
                    + source_transform.d * coords[:, 0]
                    + source_transform.e * coords[:, 1],
                )),
            )

        for feature in data["features"]["xy"]:
            subtype = feature["properties"]["subtype"]
            polygon = wkt_loads(feature["wkt"])

            if subtype == "no-damage":
                edge_color = "blue"
                intact_count += 1
            elif subtype in {"minor-damage", "major-damage", "destroyed"}:
                edge_color = "red"
                damaged_count += 1
            else:
                skipped_count += 1
                continue

            geographic_polygon = xy_to_geographic(polygon)

            projected = transform_geom(
                source_crs,
                target_crs,
                mapping(geographic_polygon),
            )

            from shapely.geometry import shape
            projected_polygon = shape(projected)

            if projected_polygon.is_empty or not projected_polygon.is_valid:
                skipped_count += 1
                continue

            # Convert projected map coordinates to Sentinel-2 pixel coordinates.
            pixel_polygon = shapely_transform(
                projected_polygon,
                lambda coords: np.column_stack(
                    (~target_transform) * (coords[:, 0], coords[:, 1])
                ),
            )

            polygons = (
                [pixel_polygon]
                if pixel_polygon.geom_type == "Polygon"
                else list(pixel_polygon.geoms)
                if pixel_polygon.geom_type == "MultiPolygon"
                else []
            )

            for part in polygons:
                coords = np.asarray(part.exterior.coords)
                ax.add_patch(
                    MplPolygon(
                        coords,
                        closed=True,
                        fill=False,
                        edgecolor=edge_color,
                        linewidth=0.6,
                    )
                )

    ax.set_xlim(0, target_width)
    ax.set_ylim(target_height, 0)
    ax.set_title(
        f"{args.uid}: xBD polygon outlines on Sentinel-2 RGB"
    )
    ax.set_axis_off()
    fig.tight_layout()

    output_path = Path("palu_polygon_alignment.png")
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.show()

    print(f"Intact polygons drawn: {intact_count}")
    print(f"Damaged polygons drawn: {damaged_count}")
    print(f"Skipped polygons: {skipped_count}")
    print(f"Diagnostic figure saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
