import cv2
import numpy as np

from .shape_class import ShapeOnMap
from .utility import Utils


class Map:

    def __init__(self, shapefile, image_size, background_color=[0, 0, 0]):
        self.shapefile = shapefile
        self.shape_dict = self.build_shape_dict(self.shapefile.df_sf)
        # different from shape_dict if filter on the base map but not on the data to plot
        self.shape_dict_filt = self.build_shape_dict(self.shapefile.df_sf)
        self.image_size = image_size
        self.max_bound = self.find_max_coords()[0]
        self.min_bound = self.find_max_coords()[1]
        self.projection = {}
        self.map_file = None
        self.background_color = background_color  # Default black background

    def build_shape_dict(self, ref_df):
        index_list = ref_df.index.tolist()
        shape_dict = {}
        for shape_id in index_list:
            shape = ShapeOnMap(self.shapefile.shapefile, shape_id)
            shape_dict[shape_id] = shape

        return shape_dict

    def find_max_coords(self):

        all_max_bound = []
        all_min_bound = []
        shape_dict = self.shape_dict
        for zone_id in shape_dict:
            zone_shape = shape_dict[zone_id]
            max_bound_zone = zone_shape.max_bound
            min_bound_zone = zone_shape.min_bound
            all_max_bound.append(max_bound_zone)
            all_min_bound.append(min_bound_zone)

        map_max_bound, unused_max = Utils.calculate_boundaries(all_max_bound)
        unused_min, map_min_bound = Utils.calculate_boundaries(all_min_bound)

        return (map_max_bound, map_min_bound)

    def render_map(self):

        # first we create a blank image, on which we will draw the base map
        width = self.image_size[0]
        height = self.image_size[1]
        # ex: size of the image 1080 height, 1920 width, 3 channels of colour
        base_map = np.zeros((height, width, 3), np.uint8)
        base_map[:, :] = self.background_color

        # we draw each shape of the dictionary on the blank image
        for shape_id in self.shape_dict_filt:
            shape = self.shape_dict_filt[shape_id]
            points = shape.points
            pts = np.array(points, np.int32)
            cv2.polylines(base_map, [pts], True, shape.color_line,
                          shape.line_thick, cv2.LINE_AA)

        self.map_file = base_map
