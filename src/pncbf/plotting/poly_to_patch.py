import numpy as np
import shapely
from matplotlib.patches import PathPatch
from matplotlib.path import Path


def poly_to_patch(poly: shapely.Polygon, **kwargs) -> PathPatch:
    ext_path = np.asarray(poly.exterior.coords)[:, :2]
    path = Path.make_compound_path(Path(ext_path), *[Path(np.asarray(ring.coords)[:, :2]) for ring in poly.interiors])

    patch = PathPatch(path, **kwargs)
    return patch
