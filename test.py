import cv2
# import numpy as np
# import pandas as pd
import shapefile as shp
import unittest


import classfile
from utility import Utils


class TestUtils(unittest.TestCase):
    def test_calculate_boundaries(self):
        """
        Test that it can calculate the min and max sets of coordinates
        """
        points = [(1, 5), (100, 150), (4, 250)]
        result = Utils.calculate_boundaries(points)
        self.assertEqual(result, (((100, 250), (1, 5))))

    def test_calculate_centroid(self):
        """
        Test that it can calculate the centroid of a list of points
        """
        points = [(1, 5), (100, 150), (4, 250)]
        result = Utils.calculate_centroid(points)
        self.assertEqual(result, (35, 135))

    def test_convert_id(self):
        """
        Test the proper conversion of an index from [1-x] to [0, x-1] interval
        """

        idx = 1
        result = Utils.convert_id(idx)
        self.assertEqual(result, 0)

    def test_convert_id_inverse(self):
        """
        Test the proper conversion of an index from [0, x-1] to [1-x] interval
        """

        inverse = True
        idx = 1
        result = Utils.convert_id(idx, inverse)
        self.assertEqual(result, 2)


class TestTextClass(unittest.TestCase):
    pass


class TestMapClass(unittest.TestCase):
    def setUp(self):
        shp_path = "./nyc_taxi_zones/taxi_zones.shp"
        self.sf = classfile.ShapeFile(shp_path)
        self.base_map = classfile.Map(self.sf, [1920, 1080])

    def test_init_map(self):
        """
        Test the proper instanciation of a Map class object
        """
        # Test the build shape_dict method
        self.assertEqual(len(self.base_map.shape_dict), 263)
        self.assertTrue(isinstance(self.base_map.shape_dict[0], classfile.ShapeOnMap))
        self.assertEqual(len(self.base_map.shape_dict_filt), 263)
        self.assertTrue(isinstance(self.base_map.shape_dict_filt[0], classfile.ShapeOnMap))
        # Test the find_max_coords method
        self.assertEqual(self.base_map.max_bound, (1067382.508405164, 272844.2940054685))
        self.assertEqual(self.base_map.min_bound, (913175.109008804, 120121.88125434518))
        self.assertEqual(self.base_map.projection, {})
        self.assertEqual(self.base_map.map_file, None)
        self.assertEqual(self.base_map.image_size, [1920, 1080])
        self.assertEqual(self.base_map.background_color, [0, 0, 0])

    def test_build_shape_dict(self):
        """
        Test the proper construction of a shape_dict from a dataframe
        Uses a shapefile provided with test.py
        """
        shape_dict = self.base_map.build_shape_dict(self.sf.df_sf)
        self.assertEqual(len(shape_dict), 263)
        self.assertTrue(isinstance(shape_dict[0], classfile.ShapeOnMap))

    def test_find_max_coords(self):
        """
        Test the proper calculation of the max boundaries of the map instance
        Uses a shapefile provided with test.py
        """
        map_max_bound, map_min_bound = self.base_map.find_max_coords()
        self.assertEqual(map_max_bound, (1067382.508405164, 272844.2940054685))
        self.assertEqual(map_min_bound, (913175.109008804, 120121.88125434518))

    def tearDown(self):
        self.sf = None
        self.base_map = None


    # def render_map(self):

    #     # first we create a blank image, on which we will draw the base map
    #     width = self.image_size[0]
    #     height = self.image_size[1]
    #     # ex: size of the image 1080 height, 1920 width, 3 channels of colour
    #     base_map = np.zeros((height, width, 3), np.uint8)
    #     base_map[:, :] = self.background_color

    #     # we draw each shape of the dictionary on the blank image
    #     for shape_id in self.shape_dict_filt:
    #         shape = self.shape_dict_filt[shape_id]
    #         points = shape.points
    #         pts = np.array(points, np.int32)
    #         cv2.polylines(base_map, [pts], True, shape.color_line,
    #                       shape.line_thick, cv2.LINE_AA)

    #     self.map_file = base_map


class TestPointClass(unittest.TestCase):
    def setUp(self):
        self.point_instance = classfile.PointOnMap([1, 1], 1, (0, 0, 0))

    def test_init_point(self):
        """
        Test the proper instanciation of the Point Class
        """
        self.assertEqual(self.point_instance.x_coord_or, 1)
        self.assertEqual(self.point_instance.y_coord_or, 1)
        self.assertEqual(self.point_instance.x_coord_curr, 1)
        self.assertEqual(self.point_instance.y_coord_curr, 1)
        self.assertEqual(self.point_instance.weight, 1)
        self.assertEqual(self.point_instance.color, (0, 0, 0))

    def test_render_point_on_map(self):
        pass

    def test_interpolate_next_position(self):
        """
        Test the proper computation of the interpolation of coordinates
        """
        target_coords = [4, 4]
        tot_frames = 4
        curr_frame = 1
        self.point_instance.interpolate_next_position(target_coords, tot_frames, curr_frame)
        self.assertEqual(self.point_instance.x_coord_curr, 2)
        self.assertEqual(self.point_instance.y_coord_curr, 2)
        self.assertEqual(self.point_instance.x_coord_or, 1)
        self.assertEqual(self.point_instance.y_coord_or, 1)

    def tearDown(self):
        self.point_instance = None


class TestProjectionClass(unittest.TestCase):
    pass


class TestShapeClass(unittest.TestCase):
    def setUp(self):
        shp_path = "./nyc_taxi_zones/taxi_zones.shp"
        self.sf = classfile.ShapeFile(shp_path)
        self.shape = classfile.ShapeOnMap(self.sf.shapefile, 0)

    def test_init_shape(self):
        self.assertEqual(self.shape.shapefile, self.sf.shapefile)
        self.assertEqual(self.shape.shape_id, 0)
        self.assertEqual(self.shape.points[0], (933100.9183527103, 192536.08569720192))
        self.assertEqual(self.shape.center, (935920.9674123558, 190798.94856132902))
        self.assertEqual(self.shape.max_bound, (941810.0086498931, 197256.2108251324))
        self.assertEqual(self.shape.min_bound, (931553.4909607167, 183788.04973023868))
        self.assertEqual(self.shape.color_line, (255, 255, 255))
        self.assertEqual(self.shape.line_thick, 1)
        self.assertEqual(self.shape.color_fill, (0, 0, 0))

    def test_get_shape_coords(self):
        points = self.shape.get_shape_coords()[0]
        center = self.shape.get_shape_coords()[1]
        max_bound = self.shape.get_shape_coords()[2]
        min_bound = self.shape.get_shape_coords()[3]
        self.assertEqual(points[0], (933100.9183527103, 192536.08569720192))
        self.assertEqual(center, (935920.9674123558, 190798.94856132902))
        self.assertEqual(max_bound, (941810.0086498931, 197256.2108251324))
        self.assertEqual(min_bound, (931553.4909607167, 183788.04973023868))

    def test_project_shape_coords(self):
        pass

    def test_fill_in_shape(self):
        pass

    def tearDown(self):
        self.sf = None
        self.shape = None

    # def project_shape_coords(self, projection):

    #     shape_zone = self.shapefile.shape(self.shape_id)
    #     points = [projection.apply_projection([i[0], i[1]]) for i in shape_zone.points]
    #     points = [projection.apply_translation([i[0], i[1]]) for i in points]
    #     self.points = points

    #     x_center, y_center = Utils.calculate_centroid(points)
    #     self.center = (x_center, y_center)

    #     max_bound, min_bound = Utils.calculate_boundaries(points)
    #     self.max_bound = max_bound
    #     self.min_bound = min_bound

    # def fill_in_shape(self, map_to_render):
    #     pts = np.array(self.points, np.int32)
    #     cv2.fillPoly(map_to_render, [pts], self.color_fill)




class TestShapefileClass(unittest.TestCase):
    def setUp(self):
        shp_path = "./nyc_taxi_zones/taxi_zones.shp"
        self.sf = classfile.ShapeFile(shp_path)

    def test_init_sf(self):
        """
        Test the proper instanciation of the Shapefile Class
        """
        # Test the shapefile Reader object
        self.assertEqual(len(self.sf.shapefile), 263)
        self.assertTrue(isinstance(self.sf.shapefile, shp.Reader))
        self.assertEqual(self.sf.shapefile.shapeTypeName, 'POINT')
        # Test the shapefile dataframe object
        self.assertEqual(self.sf.df_sf.shape, (263, 7))
        self.assertEqual(self.sf.df_sf.columns.values.tolist(), ['OBJECTID', 'Shape_Leng', 'Shape_Area', 'zone', 'LocationID', 'borough', 'coords'])
        # Test the shapefile shape_dict object
        self.assertEqual(len(self.sf.shape_dict_sf), 263)
        self.assertTrue(isinstance(self.sf.shape_dict_sf[0], classfile.ShapeOnMap))

    def test_sf_reader(self):
        """
        Test the instanciation of the Reader class from a shapefile
        """
        # Test the shapefile Reader object
        self.assertEqual(len(self.sf.shapefile), 263)
        self.assertTrue(isinstance(self.sf.shapefile, shp.Reader))
        self.assertEqual(self.sf.shapefile.shapeTypeName, 'POINT')

    def test_shp_to_df(self):
        """
        Test the proper construction of a dataframe from a shapefile
        """
        df_sf = self.sf.shp_to_df()
        self.assertEqual(df_sf.shape, (283, 5))
        self.assertEqual(df_sf.columns.values.tolist(), ['OBJECTID', 'Shape_Leng', 'Shape_Area', 'zone', 'LocationID', 'borough', 'coords'])

    def test_build_shape_dict(self):
        """
        Test the proper construction of a shape_dict from a dataframe
        """
        df_sf = self.sf.shp_to_df()
        shape_dict_sf = self.sf.build_shape_dict(df_sf)
        self.assertEqual(len(shape_dict_sf), 283)
        self.assertTrue(isinstance(shape_dict_sf[0], classfile.ShapeOnMap))

    def test_filter_shape_to_render_str(self):
        """
        Test the proper filtering of a dataframe when the condition statement is a string
        """
        cond_stat = 'Newark Airport'
        attr = 'zone'
        df_filtered = self.sf.filter_shape_to_render(cond_stat, attr)
        self.assertEqual(df_filtered.shape, (1, 7))
        self.assertEqual(df_filtered.iloc[0, [0, 1, 2, 3, 4]].tolist(), [1, 0.116357, 0.000782307, 'Newark Airport', 1, 'EWR'])

    def test_filter_shape_to_render_arr(self):
        """
        Test the proper filtering of a dataframe when the condition statement is an array
        """
        cond_stat = ['Newark Airport', 'Jamaica Bay']
        attr = 'zone'
        df_filtered = self.sf.filter_shape_to_render(cond_stat, attr)
        self.assertEqual(df_filtered.shape, (2, 7))
        self.assertEqual(df_filtered.iloc[0, [0, 1, 2, 3, 4]].tolist(), [1, 0.116357, 0.000782307, 'Newark Airport', 1, 'EWR'])
        self.assertEqual(df_filtered.iloc[1, [0, 1, 2, 3, 4]].tolist(), [2, 0.43347, 0.00486634, 'Jamaica Bay', 2, 'Queens'])

    def test_filter_shape_to_render_error_cond(self):
        """
        Test the proper filtering of a dataframe when the condition statement is invalid
        """
        cond_stat = 'unexisting_value'
        attr = 'zone'
        #df_filtered = self.sf.filter_shape_to_render(cond_stat, attr)
        pass

    def test_filter_shape_to_render_error_attr(self):
        """
        Test the proper filtering of a dataframe when the attribute statement is invalid
        """
        cond_stat = 'Alphabet City'
        attr = 'name'
        #df_filtered = self.sf.filter_shape_to_render(cond_stat, attr)
        pass

    def tearDown(self):
        self.sf = None


    # def filter_shape_to_render(self, cond_stat, attr):
    #     # cond_stat in the form of a string or an array
    #     # attr in the form of a str, a column name of df
    #     try:
    #         if type(cond_stat) == str:
    #             filtered_df = self.df_sf[self.df_sf[attr] == cond_stat]
    #         elif type(cond_stat) == list:
    #             filtered_df = self.df_sf[self.df_sf[attr].isin(cond_stat)]

    #         return filtered_df
    #     except:
             # print("""Error parsing condition statement or attribute
             #        Condition statement must be str or arr of values in the shapefile_dataframe
             #        Attribute must be a column name of the shapefile_dataframe
             #        See ShapeFile().df_sf for more details
             #        """)
    



if __name__ == '__main__':
    unittest.main()

# mock, spies
# import the libraries
# find max coords
# define projection
# projection with margin
# apply projection/translation
# render a map from a shapefile (call the right method with the right parameters)
# draw a point on map
# put text on map
# fill in shape



#Code changes - improvements
# convert id should not go beyond 0
# error message if points in boundaries and centroid are not lists of tuples
# format of input for position interpolation

