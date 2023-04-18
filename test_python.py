# [0.5, 0.3, 0.7, 0.3, 0.7, 0.5, 0.5, 0.5]
import numpy as np
from shapely.geometry import Point, Polygon


polygons = [0.5, 0.3, 0.7, 0.3, 0.7, 0.5, 0.5, 0.5]


polygons = np.array(polygons).reshape(-1,2).tolist()
print(polygons)
