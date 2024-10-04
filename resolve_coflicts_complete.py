import os
import re
import argparse
import logging


import geopandas
import pandas as pd
import numpy as np
from shapely import geometry, make_valid, unary_union

from sopa.io.standardize import read_zarr_standardized
from sopa._sdata import get_spatial_image

from spatialdata.models import ShapesModel
from spatialdata.transformations import get_transformation

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def remove_edge_polys(
    df, base_polygon, buffer_outer=10, buffer_inner=5, geometry_field: str = "geometry"
):
    # Create crop frame based on the supplied polygon
    crop_frame = base_polygon.buffer(buffer_outer).difference(
        base_polygon.buffer(-buffer_inner)
    )

    # List to store indices to remove
    indices_to_remove = []

    for idx in df.index.unique():
        cur_geom = df.loc[idx, geometry_field]
        if cur_geom.intersects(crop_frame):
            indices_to_remove.append(idx)

    # Remove all rows for the identified indices
    df = df.drop(indices_to_remove)

    return df


def largest_geometry(shape):
    """
    If passed a Polygon, returns the Polygon.
    If passed a MultiPolygon, returns the largest Polygon region.
    Else throws TypeError
    """
    if type(shape) is geometry.Polygon:
        return shape
    elif type(shape) is geometry.MultiPolygon:
        if len(shape.geoms) == 0:
            print("Empty multipolygon passed, an empty polygon will be returned")
            return geometry.Polygon()
        sizes = np.array([(idx, g.area) for idx, g in enumerate(shape.geoms)])
        idx_largest = sizes[sizes[:, 1].argmax(), 0]
        return shape.geoms[int(idx_largest)]
    else:
        raise TypeError(f"Objects of type {type(shape)} are not supported")


def smooth_and_simplify(poly, radius, tol):
    if isinstance(poly, geometry.MultiPolygon):
        buffered_shapes = (
            p.buffer(-radius).buffer(radius * 2).buffer(-radius) for p in poly.geoms
        )
        buffered_multipolygons = (
            p if type(p) is geometry.MultiPolygon else geometry.MultiPolygon([p])
            for p in buffered_shapes
        )
        buffered_polygons = (p for mp in buffered_multipolygons for p in mp.geoms)
        poly = geometry.MultiPolygon(buffered_polygons)
    elif isinstance(poly, geometry.Polygon):
        poly = poly.buffer(-radius).buffer(radius * 2).buffer(-radius)
        return poly
    return largest_geometry(poly.simplify(tolerance=tol))


def convert_to_multipoly(shape):
    if type(shape) is geometry.Polygon:
        return geometry.multipolygon.MultiPolygon([shape])
    elif type(shape) is geometry.MultiPolygon:
        return shape
    elif type(shape) is geometry.GeometryCollection:
        poly_shapes = [
            g
            for g in shape.geoms
            if type(g) in [geometry.Polygon, geometry.MultiPolygon]
        ]
        poly = get_valid_geometry(unary_union(poly_shapes))
        return (
            poly
            if type(poly) is geometry.MultiPolygon
            else geometry.multipolygon.MultiPolygon([poly])
        )
    else:
        # If type is not Polygon or Multipolygon, the shape
        # is strange / small and should be rejected
        return geometry.MultiPolygon()


def combine_segmentations(segmentations):
    non_empty_segmentations = [seg for seg in segmentations if len(seg) > 0]
    if len(non_empty_segmentations) > 1:
        to_concat = [seg for seg in non_empty_segmentations]
        df = geopandas.GeoDataFrame(pd.concat(to_concat, ignore_index=True))
        print(f"Combined length {len(df)}")
        return df
    elif len(non_empty_segmentations) == 1:
        return non_empty_segmentations[0]
    else:
        return segmentations[0] if len(segmentations) > 0 else geopandas.GeoDataFrame()


def get_valid_geometry(shape):
    try:
        valid_shape = make_valid(shape)
        if isinstance(valid_shape, geometry.MultiPolygon):
            # Choose the largest polygon by area, or just the first one
            largest_polygon = max(valid_shape.geoms, key=lambda p: p.area)
            return largest_polygon
        elif isinstance(valid_shape, geometry.Polygon):
            return valid_shape
        else:
            print("Unexpected geometry type, converting to empty polygon.")
            return geometry.Polygon()
    except ValueError:
        print("Entity could not be converted to a valid polygon, removing.")
        return geometry.Polygon()


def trim_large_cells(combined, large_cell, small_cell, min_distance: int):
    """
    Trims area from larger entity
    """

    try:
        # Trims the larger geometry with a small buffer
        valid_large = smooth_and_simplify(
            get_valid_geometry(combined.loc[large_cell, "geometry"]), 0.5, 2
        )
        valid_small = smooth_and_simplify(
            get_valid_geometry(combined.loc[small_cell, "geometry"]), 0.5, 2
        )
        trimmed_raw = valid_large.difference(valid_small.buffer(min_distance))

        trimmed_geometry = get_valid_geometry(trimmed_raw)
    except ValueError:
        print("Entity could not be converted to a valid polygon, removing.")
        trimmed_geometry = geometry.Polygon()

    # Overwrites large geometry with trimmed geometry
    combined.loc[large_cell, "geometry"] = trimmed_geometry


def extract_number(file_path):
    match = re.search(r"(\d+)\.parquet", file_path)
    return int(match.group(1)) if match else 0


def add_shape_df(sdata, geo_df, image_key, shapes_key):
    image = get_spatial_image(sdata, image_key)
    geo_df.index = image_key + geo_df.index.astype(str)
    geo_df = ShapesModel.parse(
        geo_df, transformations=get_transformation(image, get_all=True).copy()
    )
    sdata.shapes[shapes_key] = geo_df
    if sdata.is_backed():
        sdata.write_element(shapes_key, overwrite=True)
    log.info(f"Added {len(geo_df)} cell boundaries in sdata['{shapes_key}']")


def main():
    parser = argparse.ArgumentParser(description="Resolve conflicts in segmentation")
    parser.add_argument("sdata", type=str, help="Spatial Data object being assembled")
    parser.add_argument(
        "--patch-dir",
        type=str,
        help="Folder containing the segmentation files",
        required=True,
    )
    args = parser.parse_args()

    sdata = read_zarr_standardized(args.sdata)
    # image_key = get_key(sdata, "images")
    image_key = "images"

    # all parquet files in the folder ascending order
    test_fovs = sorted(
        [f for f in os.listdir(args.patch_dir) if f.endswith(".parquet")],
        key=extract_number,
    )
    test_fovs = [os.path.join(args.patch_dir, f) for f in test_fovs]

    fovs = []
    for fov, patch in zip(test_fovs, sdata.shapes["sopa_patches"].geometry):
        seg = geopandas.read_parquet(fov)
        seg = remove_edge_polys(seg, patch)
        fovs.append(seg)

    combined = combine_segmentations(fovs)
    log.info(f"Processing {len(combined)} polygons")

    overlaps = combined.sindex.query(combined.geometry, predicate="intersects").T
    overlaps = np.array([pair for pair in overlaps if pair[0] != pair[1]])

    log.info(f"Found {len(overlaps)} uncaught overlaps")

    bad_polygons = []

    # overlaps list of entity ids or index in our case
    for problem in overlaps:
        entity_id_left, entity_id_right = problem[0], problem[1]
        if entity_id_left in bad_polygons or entity_id_right in bad_polygons:
            continue
        # get area from geopandas dataframe by index
        area_left = combined.loc[entity_id_left, "geometry"].area
        area_right = combined.loc[entity_id_right, "geometry"].area
        intersection = (
            combined.loc[entity_id_left, "geometry"]
            .intersection(combined.loc[entity_id_right, "geometry"])
            .area
        )
        intersection_pct = intersection / min(area_left, area_right)
        # If overlap is > 50% of either cell, eliminate the small cell and keep the big one
        if intersection_pct > 0.5:
            if area_left > area_right:
                bad_polygons.append(entity_id_right)
            else:
                bad_polygons.append(entity_id_left)

    # With large overlaps resolved, re-identify problem sets and trim overlaps

    combined = combined.drop(bad_polygons)
    combined = combined.reset_index(drop=True)

    overlaps = combined.sindex.query(combined.geometry, predicate="intersects").T
    overlaps = np.array([pair for pair in overlaps if pair[0] != pair[1]])

    log.info(f"After resolution step 1, {len(overlaps)} overlapping polygons remain")

    for problem in overlaps:
        entity_id_left, entity_id_right = problem[0], problem[1]
        # get area from geopandas dataframe by index
        area_left = combined.loc[entity_id_left, "geometry"].area
        area_right = combined.loc[entity_id_right, "geometry"].area
        # Trim the larger cell to dodge the smaller cell
        if area_left > area_right:
            trim_large_cells(combined, entity_id_left, entity_id_right, 1)
        else:
            trim_large_cells(combined, entity_id_left, entity_id_right, 1)

    # remove polygons smaller than 500
    combined = combined[combined.geometry.area > 500]

    # After both steps, check for any remaining overlaps
    overlaps = combined.sindex.query(combined.geometry, predicate="intersects").T
    overlaps = np.array([pair for pair in overlaps if pair[0] != pair[1]])

    log.info(f"After both resolution steps, found {len(overlaps)} uncaught overlaps")

    add_shape_df(sdata, combined, image_key, "cellpose_boundaries")


if __name__ == "__main__":
    main()
