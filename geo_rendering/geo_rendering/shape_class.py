import cv2
import numpy as np

from .utility import Utils


class ShapeOnMap:

    def __init__(self, shapefile, shape_id):
        self.shapefile = shapefile
        self.shape_id = shape_id
        self.points = self.get_shape_coords()[0]
        self.center = self.get_shape_coords()[1]
        self.max_bound = self.get_shape_coords()[2]
        self.min_bound = self.get_shape_coords()[3]
        self.color_line = (255, 255, 255)  # Default white line
        self.line_thick = 1
        self.color_fill = (0, 0, 0)  # Default black fill

    def get_shape_coords(self):

        shape_zone = self.shapefile.shape(self.shape_id)
        points = [(i[0], i[1]) for i in shape_zone.points]
        x_center, y_center = Utils.calculate_centroid(points)
        center = (x_center, y_center)
        max_bound, min_bound = Utils.calculate_boundaries(points)

        return (points, center, max_bound, min_bound)

    def project_shape_coords(self, projection):

        shape_zone = self.shapefile.shape(self.shape_id)
        points = [projection.apply_projection([i[0], i[1]]) for i in shape_zone.points]
        points = [projection.apply_translation([i[0], i[1]]) for i in points]
        self.points = points

        x_center, y_center = Utils.calculate_centroid(points)
        self.center = (x_center, y_center)

        max_bound, min_bound = Utils.calculate_boundaries(points)
        self.max_bound = max_bound
        self.min_bound = min_bound

    def fill_in_shape(self, map_to_render):
        pts = np.array(self.points, np.int32)
        cv2.fillPoly(map_to_render, [pts], self.color_fill)
