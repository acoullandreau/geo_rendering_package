import pandas as pd
import shapefile as shp
from utility import Utils


class Map:

    def __init__(self, shapefile, image_size):
        self.shapefile = shapefile
        self?image_size = image_size
        self.max_bound = -99999999
        self.min_bound = 99999999

    def find_max_coords(self):

        all_max_bound = []
        all_min_bound = []
        shape_dict = self.shapefile.shape_dict

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
