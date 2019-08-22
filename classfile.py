import cv2
import numpy as np
import pandas as pd
import shapefile as shp

from utility import Utils


class Map:

    def __init__(self, shapefile, image_size):
        self.shapefile = shapefile
        self.image_size = image_size
        self.max_bound = -99999999
        self.min_bound = 99999999
        self.projection = {}
        self.sf_shape_dict = self.shapefile.shape_dict
        self.shape_dict = {}
        self.map_file = None

    def find_max_coords(self):

        all_max_bound = []
        all_min_bound = []
        shape_dict = self.sf_shape_dict

        for zone in shape_dict:
            zone_shape = shape_dict[zone]
            max_bound_zone = zone_shape['max_bound']
            min_bound_zone = zone_shape['min_bound']
            all_max_bound.append(max_bound_zone)
            all_min_bound.append(min_bound_zone)

        map_max_bound, unused_max = Utils.calculate_boundaries(all_max_bound)
        unused_min, map_min_bound = Utils.calculate_boundaries(all_min_bound)

        self.max_bound = map_max_bound
        self.min_bound = map_min_bound

    def define_projection(self):

        # We get the max 'coordinates' for both the target image and
        # the shape we want to draw
        image_x_max = self.image_size[0]
        image_y_max = self.image_size[1]
        map_x_max = self.max_bound[0]
        map_y_max = self.max_bound[1]
        map_x_min = self.min_bound[0]
        map_y_min = self.min_bound[1]

        # we check which size is bigger to know based on which axis we want
        # to scale our shape to
        # we do the comparison using the aspect ratio expectations (dividing
        # each axis by the size of the target axis in the new scale)
        ratio_x = (map_x_max - map_x_min)/image_x_max
        ratio_y = (map_y_max - map_y_min)/image_y_max
        if ratio_x > ratio_y:
            conversion = image_x_max / (map_x_max - map_x_min)
            axis_to_center = 'y'  # we store the axis we will want to center on
            # based on which axis we perform the scaling from
        else:
            conversion = image_y_max / (map_y_max - map_y_min)
            axis_to_center = 'x'

        self.projection['image_size'] = self.image_size
        self.projection['map_max_bound'] = self.max_bound
        self.projection['map_min_bound'] = self.min_bound
        self.projection['conversion'] = conversion
        self.projection['axis_to_center'] = axis_to_center

    def apply_projection(self, x, y, inverse=False):

        x_min = self.projection['map_min_bound'][0]
        y_min = self.projection['map_min_bound'][1]
        conversion = self.projection['conversion']

        if inverse is False:
            # to be able to center the image, we first translate the
            # coordinates to the origin
            x = (x - x_min) * conversion
            y = (y - y_min) * conversion
        else:
            x = (x + x_min) / conversion
            y = (y + y_min) / conversion

        return x, y

    def apply_project_to_shape_dict(self):

        proj = self.projection
        axis_to_center = proj['axis_to_center']
        image_x_max = proj['image_size'][0]
        image_y_max = proj['image_size'][1]
        map_x_max = proj['map_max_bound'][0]
        map_y_max = proj['map_max_bound'][1]
        map_max_converted = (self.apply_projection(map_x_max, map_y_max, proj))
        map_x_min = proj['map_min_bound'][0]
        map_y_min = proj['map_min_bound'][1]
        map_min_converted = (self.apply_projection(map_x_min, map_y_min, proj))

        if axis_to_center == 'x':
            map_x_max_conv = map_max_converted[0]
            map_x_min_conv = map_min_converted[0]
            center_translation = (image_x_max - (map_x_max_conv - map_x_min_conv))/2
        else:
            map_y_max_conv = map_max_converted[1]
            map_y_min_conv = map_min_converted[1]
            center_translation = (image_y_max - (map_y_max_conv - map_y_min_conv))/2

        for zone_id in self.shape_dict:
            curr_shape = self.shape_dict[zone_id]

            points = curr_shape['points']
            x_center = curr_shape['center'][0]
            y_center = curr_shape['center'][1]
            max_bound = curr_shape['max_bound']
            min_bound = curr_shape['min_bound']

            converted_points = []
            for point in points:
                # we convert the coordinates to the new coordinate system
                conv_point = [0, 0]
                conv_point[0], conv_point[1] = self.apply_projection(point[0], point[1], proj)
                # we center the map on the axis that was not used to scale the image
                if axis_to_center == 'x':
                    conv_point[0] = conv_point[0] + center_translation
                else:
                    conv_point[1] = conv_point[1] + center_translation

                # we mirror the image to match the axis alignment
                conv_point[1] = image_y_max - conv_point[1]
                converted_points.append(conv_point)

            # we convert the center and the max and min boundaries
            x_center, y_center = Utils.calculate_centroid(converted_points)
            max_bound = (self.apply_projection(max_bound[0], max_bound[1], proj))
            min_bound = (self.apply_projection(min_bound[0], min_bound[1], proj))

            # We edit the dictionary with the new coordinates
            self.shape_dict[zone_id] = {}
            self.shape_dict[zone_id]['points'] = converted_points
            self.shape_dict[zone_id]['center'] = (x_center, y_center)
            self.shape_dict[zone_id]['max_bound'] = max_bound
            self.shape_dict[zone_id]['min_bound'] = min_bound

    def draw_base_map(self):

        # first we create a blank image, on which we will draw the base map
        width = self.image_size[0]
        height = self.image_size[1]
        # ex: size of the image 1080 height, 1920 width, 3 channels of colour
        base_map = np.zeros((height, width, 3), np.uint8)
        base_map[:, :] = [0, 0, 0]  # Sets the color to white

        # we draw each shape of the dictionary on the blank image
        for item in self.shape_dict:
            shape = self.shape_dict[item]
            points = shape['points']
            pts = np.array(points, np.int32)
            cv2.polylines(base_map, [pts], True, (255, 255, 255), 1, cv2.LINE_AA)

        self.map_file = base_map


class ShapeFile:

    def __init__(self, file_path):
        self.path = file_path
        self.shapefile = None
        self.df_sf = None
        self.shape_dict = None

    def sf_reader(self, path):
        shapefile = shp.Reader(self.path)
        self.shapefile = shapefile

    def shp_to_df(self):
        sf = self.shapefile
        fields = [x[0] for x in sf.fields][1:]
        records = sf.records()
        shps = [s.points for s in sf.shapes()]
        df = pd.DataFrame(columns=fields, data=records)
        df = df.assign(coords=shps)
        self.df_sf = df

    def process_shape_boundaries(self):

        shape_dict = {}
        index_list = self.df_sf.index.tolist()

        for zone_id in index_list:
            # for each zone id available in the shapefile
            if zone_id not in shape_dict:
                # we only process the coordinates if it is not yet in the dict
                shape_zone = self.shapefile.shape(zone_id)

                points = [(i[0], i[1]) for i in shape_zone.points]

                x_center, y_center = Utils.calculate_centroid(points)
                max_bound, min_bound = Utils.calculate_boundaries(points)

                # we add to the dict, for the zone id, the shape boundaries
                # and the coordinates of the center of the shape
                # and the zone extreme boundaries
                shape_dict[zone_id] = {}
                shape_dict[zone_id]['points'] = points
                shape_dict[zone_id]['center'] = (x_center, y_center)
                shape_dict[zone_id]['max_bound'] = max_bound
                shape_dict[zone_id]['min_bound'] = min_bound

        self.shape_dict = shape_dict


class ContextualText:

    def __init__(self, content):
        self.text_content = content
