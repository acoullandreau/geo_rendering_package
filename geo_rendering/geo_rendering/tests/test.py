import cv2
import numpy as np
import shapefile as shp
import unittest
from unittest.mock import MagicMock

from ..contextual_text_class import ContextualText
from ..map_class import Map
from ..point_class import PointOnMap
from ..projection_class import Projection
from ..shape_class import ShapeOnMap
from ..shapefile_class import ShapeFile
from ..utility import Utils


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

    @classmethod
    def setUpClass(self):
        self.text = ContextualText('This is a test', (50, 50), (255, 255, 255))
        self.cv2_original = cv2.putText
        cv2.putText = MagicMock()

    def test_init_text(self):
        """
        Test the proper instanciation of a ContextualText class object
        """
        self.assertEqual(self.text.text_content, 'This is a test')
        self.assertEqual(self.text.position, (50, 50))
        self.assertEqual(self.text.color, (255, 255, 255))
        self.assertEqual(self.text.font, cv2.FONT_HERSHEY_SIMPLEX)
        self.assertEqual(self.text.font_style, cv2.LINE_AA)
        self.assertEqual(self.text.text_size, 1)
        self.assertEqual(self.text.thickness, 1)
        self.text.text_size = 10
        self.assertEqual(self.text.text_size, 10)
        self.text.thickness = 2
        self.assertEqual(self.text.thickness, 2)

    def test_display_text(self):
        map_to_edit = 'Test Map'
        text = self.text.text_content
        pos = self.text.position
        col = self.text.color
        font = self.text.font
        size = self.text.text_size
        thick = self.text.thickness
        style = self.text.font_style
        cv2.putText(map_to_edit, text, pos, font, size, col, thick, style)
        # assertion statement
        cv2.putText.assert_called_once_with('Test Map', 'This is a test', (50, 50), 0, 1, (255, 255, 255), 1, 16)

    @classmethod
    def tearDownClass(self):
        self.text = None
        cv2.putText = self.cv2_original


class TestMapClass(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        shp_path = "geo_rendering/tests/nyc_taxi_zones/taxi_zones.shp"
        self.sf = ShapeFile(shp_path)
        self.base_map = Map(self.sf, [1920, 1080])
        self.cv2_original = cv2.polylines
        cv2.polylines = MagicMock()

    def test_init_map(self):
        """
        Test the proper instanciation of a Map class object
        """
        # Test the build shape_dict method
        self.assertEqual(len(self.base_map.shape_dict), 263)
        self.assertTrue(isinstance(self.base_map.shape_dict[0], ShapeOnMap))
        self.assertEqual(len(self.base_map.shape_dict_filt), 263)
        self.assertTrue(isinstance(self.base_map.shape_dict_filt[0], ShapeOnMap))
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
        self.assertTrue(isinstance(shape_dict[0], ShapeOnMap))

    def test_find_max_coords(self):
        """
        Test the proper calculation of the max boundaries of the map instance
        Uses a shapefile provided with test.py
        """
        map_max_bound, map_min_bound = self.base_map.find_max_coords()
        self.assertEqual(map_max_bound, (1067382.508405164, 272844.2940054685))
        self.assertEqual(map_min_bound, (913175.109008804, 120121.88125434518))

    def test_render_map(self):
        """
        Test the proper rendering of the shapes on a map file using opencv
        """
        self.base_map.render_map()
        # we ensure that the map_file is a numpy array
        self.assertTrue(isinstance(self.base_map.map_file, np.ndarray))
        self.assertEqual(self.base_map.map_file.shape[0], self.base_map.image_size[1])
        self.assertEqual(self.base_map.map_file.shape[1], self.base_map.image_size[0])
        self.assertEqual(self.base_map.map_file.shape[2], 3)  # 3 channels of color
        # we ensure that cv2.polylines was called as many times as the size of the shape_dict
        self.assertEqual(cv2.polylines.call_count, len(self.base_map.shape_dict))
        # we ensure that the arguments passed to call cv2.polylines are the expected ones
        self.assertEqual(type(cv2.polylines.call_args[0][0]), np.ndarray)
        self.assertEqual(type(cv2.polylines.call_args[0][1]), list)
        self.assertEqual(cv2.polylines.call_args[0][2], True)
        self.assertEqual(cv2.polylines.call_args[0][3], self.base_map.shape_dict[0].color_line)
        self.assertEqual(cv2.polylines.call_args[0][4], self.base_map.shape_dict[0].line_thick)
        self.assertEqual(cv2.polylines.call_args[0][5], 16)

    @classmethod
    def tearDownClass(self):
        self.sf = None
        self.base_map = None
        cv2.polylines = self.cv2_original


class TestPointClass(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.point_instance = PointOnMap([1, 1], 1, (0, 0, 0))
        self.cv2_original = cv2.circle
        cv2.circle = MagicMock()

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
        map_to_render = 'Test Map'
        self.point_instance.render_point_on_map(map_to_render)
        # assertion statement
        cv2.circle.assert_called_once_with('Test Map',
                                           (self.point_instance.x_coord_curr, self.point_instance.y_coord_curr),
                                           self.point_instance.weight,
                                           self.point_instance.color, -1)

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

    @classmethod
    def tearDownClass(self):
        self.point_instance = None
        cv2.circle = self.cv2_original


class TestProjectionClassWithMargin(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        shp_path = "geo_rendering/tests/nyc_taxi_zones/taxi_zones.shp"
        self.sf = ShapeFile(shp_path)
        self.base_map = Map(self.sf, [1920, 1080])
        self.projection = Projection(self.base_map, [10, 20, 30, 40])

    def test_0_init_projection(self):
        """
        Test the proper instanciation of the Projection Class with margins
        """
        self.assertEqual(self.projection.image_size, [1920, 1080])
        self.assertEqual(self.projection.map_max_bound, (1067382.508405164, 272844.2940054685))
        self.assertEqual(self.projection.map_min_bound, (913175.109008804, 120121.88125434518))
        self.assertEqual(self.projection.margin, [10, 20, 30, 40])
        self.assertEqual(self.projection.conversion, 0.006809740504131412)
        self.assertEqual(self.projection.axis_to_center, 'x')

    def test_define_projection(self):
        """
        Test the proper calculation of a projection conversion rate and axis to center on
        """
        conversion, axis_to_center = self.projection.define_projection()
        self.assertEqual(conversion, 0.006809740504131412)
        self.assertEqual(axis_to_center, 'x')

    def test_apply_projection(self):
        """
        Test the proper conversion of coordinates to a new projection system
        By opposition to the test test_apply_projection_inverse
        """
        coords = (1, 1)
        new_coords = self.projection.apply_projection(coords, inverse=False)
        self.assertEqual(new_coords[0], -6218.478717441366)
        self.assertEqual(new_coords[1], -817.992030469674)

    def test_apply_projection_inverse(self):
        """
        Test the proper conversion of coordinates to a "previous" projection system
        By opposition to the test test_apply_projection
        """
        coords = (-6218.478717441366, -817.992030469674)
        new_coords = self.projection.apply_projection(coords, inverse=True)
        self.assertEqual(new_coords[0], 1.0)
        self.assertEqual(new_coords[1], 1.0)

    def test_apply_translation_x_center(self):
        """
        Test the proper translation of coordinates on the x-axis
        """
        self.projection.axis_to_center = 'x'
        coords = [1, 1]
        new_coords = self.projection.apply_translation(coords)
        self.assertEqual(new_coords[0], 405.94381314691896)
        self.assertEqual(new_coords[1], 1049)

    def test_apply_translation_y_center(self):
        """
        Test the proper translation of coordinates on the y-axis
        """
        self.projection.axis_to_center = 'y'
        coords = [1, 1]
        new_coords = self.projection.apply_translation(coords)
        self.assertEqual(new_coords[0], 1)
        self.assertEqual(new_coords[1], 1049)

    @classmethod
    def tearDownClass(self):
        self.sf = None
        self.base_map = None
        self.projection = None


class TestProjectionClassWithoutMargin(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        shp_path = "geo_rendering/tests/nyc_taxi_zones/taxi_zones.shp"
        self.sf = ShapeFile(shp_path)
        self.base_map = Map(self.sf, [1920, 1080])
        self.projection = Projection(self.base_map)

    def test_0_init_projection(self):
        """
        Test the proper instanciation of the Projection Class without margins
        """
        self.assertEqual(self.projection.image_size, [1920, 1080])
        self.assertEqual(self.projection.map_max_bound, (1067382.508405164, 272844.2940054685))
        self.assertEqual(self.projection.map_min_bound, (913175.109008804, 120121.88125434518))
        self.assertEqual(self.projection.margin, [0, 0, 0, 0])
        self.assertEqual(self.projection.conversion, 0.007071653600444159)
        self.assertEqual(self.projection.axis_to_center, 'x')

    def test_define_projection(self):
        """
        Test the proper calculation of a projection conversion rate and axis to center on
        """
        conversion, axis_to_center = self.projection.define_projection()
        self.assertEqual(conversion, 0.007071653600444159)
        self.assertEqual(axis_to_center, 'x')

    def test_apply_projection(self):
        """
        Test the proper conversion of coordinates to a new projection system
        By opposition to the test test_apply_projection_inverse
        """
        coords = [1, 1]
        new_coords = self.projection.apply_projection(coords, inverse=False)
        self.assertEqual(new_coords[0], -6457.650975804495)
        self.assertEqual(new_coords[1],  -849.4532624108153)

    def test_apply_projection_inverse(self):
        """
        Test the proper conversion of coordinates to a "previous" projection system
        By opposition to the test test_apply_projection
        """
        coords = [-6457.650975804495, -849.4532624108153]
        new_coords = self.projection.apply_projection(coords, inverse=True)
        self.assertEqual(new_coords[0], 1.0)
        self.assertEqual(new_coords[1], 1.0)

    def test_apply_translation_x_center(self):
        """
        Test the proper translation of coordinates on the x-axis
        """
        self.projection.axis_to_center = 'x'
        coords = [1, 1]
        new_coords = self.projection.apply_translation(coords)
        self.assertEqual(new_coords[0], 415.7493444218005)
        self.assertEqual(new_coords[1], 1079)

    def test_apply_translation_y_center(self):
        """
        Test the proper translation of coordinates on the y-axis
        """
        self.projection.axis_to_center = 'y'
        coords = [1, 1]
        new_coords = self.projection.apply_translation(coords)
        self.assertEqual(new_coords[0], 1)
        self.assertEqual(new_coords[1], 1079)

    @classmethod
    def tearDownClass(self):
        self.sf = None
        self.base_map = None
        self.projection = None


class TestProjectionClassWithScaleAndMargin(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        shp_path = "geo_rendering/tests/nyc_taxi_zones/taxi_zones.shp"
        self.sf = ShapeFile(shp_path)
        self.base_map = Map(self.sf, [1920, 1080])
        self.projection = Projection(self.base_map, scale_param=[0.5, 0.75], margin=[10, 20, 30, 40])

    def test_0_init_projection(self):
        """
        Test the proper instanciation of the Projection Class with margins
        """
        self.assertEqual(self.projection.image_size, [1920, 1080])
        self.assertEqual(self.projection.map_max_bound, (1067382.508405164, 272844.2940054685))
        self.assertEqual(self.projection.map_min_bound, (913175.109008804, 120121.88125434518))
        self.assertEqual(self.projection.margin, [10, 20, 30, 40])
        self.assertEqual(self.projection.scale_param, [0.5, 0.75])
        self.assertEqual(self.projection.conversion, 0.006809740504131412)
        self.assertEqual(self.projection.axis_to_center, 'x')

    def test_define_projection(self):
        """
        Test the proper calculation of a projection conversion rate and axis to center on
        """
        conversion, axis_to_center = self.projection.define_projection()
        self.assertEqual(conversion, 0.006809740504131412)
        self.assertEqual(axis_to_center, 'x')

    def test_apply_projection(self):
        """
        Test the proper conversion of coordinates to a new projection system
        By opposition to the test test_apply_projection_inverse
        """
        coords = (1, 1)
        new_coords = self.projection.apply_projection(coords, inverse=False)
        self.assertEqual(new_coords[0], -3109.239358720683)
        self.assertEqual(new_coords[1], -613.4940228522555)

    def test_apply_projection_inverse(self):
        """
        Test the proper conversion of coordinates to a "previous" projection system
        By opposition to the test test_apply_projection
        """
        coords = (-3109.239358720683, -613.4940228522555)
        new_coords = self.projection.apply_projection(coords, inverse=True)
        self.assertEqual(new_coords[0], 1.0)
        self.assertEqual(new_coords[1], 1.0)

    def test_apply_translation_x_center(self):
        """
        Test the proper translation of coordinates on the x-axis
        """
        self.projection.axis_to_center = 'x'
        coords = [1, 1]
        new_coords = self.projection.apply_translation(coords)
        self.assertEqual(new_coords[0], 668.4719065734595)
        self.assertEqual(new_coords[1], 1049)

    def test_apply_translation_y_center(self):
        """
        Test the proper translation of coordinates on the y-axis
        """
        self.projection.axis_to_center = 'y'
        coords = [1, 1]
        new_coords = self.projection.apply_translation(coords)
        self.assertEqual(new_coords[0], 1)
        self.assertEqual(new_coords[1], 919)

    @classmethod
    def tearDownClass(self):
        self.sf = None
        self.base_map = None
        self.projection = None


class TestProjectionClassWithScaleWithoutMargin(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        shp_path = "geo_rendering/tests/nyc_taxi_zones/taxi_zones.shp"
        self.sf = ShapeFile(shp_path)
        self.base_map = Map(self.sf, [1920, 1080])
        self.projection = Projection(self.base_map, scale_param=[0.5, 0.75])

    def test_0_init_projection(self):
        """
        Test the proper instanciation of the Projection Class without margins
        """
        self.assertEqual(self.projection.image_size, [1920, 1080])
        self.assertEqual(self.projection.map_max_bound, (1067382.508405164, 272844.2940054685))
        self.assertEqual(self.projection.map_min_bound, (913175.109008804, 120121.88125434518))
        self.assertEqual(self.projection.margin, [0, 0, 0, 0])
        self.assertEqual(self.projection.scale_param, [0.5, 0.75])
        self.assertEqual(self.projection.conversion, 0.007071653600444159)
        self.assertEqual(self.projection.axis_to_center, 'x')

    def test_define_projection(self):
        """
        Test the proper calculation of a projection conversion rate and axis to center on
        """
        conversion, axis_to_center = self.projection.define_projection()
        self.assertEqual(conversion, 0.007071653600444159)
        self.assertEqual(axis_to_center, 'x')

    def test_apply_projection(self):
        """
        Test the proper conversion of coordinates to a new projection system
        By opposition to the test test_apply_projection_inverse
        """
        coords = (1, 1)
        new_coords = self.projection.apply_projection(coords, inverse=False)
        self.assertEqual(new_coords[0], -3228.8254879022475)
        self.assertEqual(new_coords[1], -637.0899468081116)

    def test_apply_projection_inverse(self):
        """
        Test the proper conversion of coordinates to a "previous" projection system
        By opposition to the test test_apply_projection
        """
        coords = (-3228.8254879022475, -637.0899468081116)
        new_coords = self.projection.apply_projection(coords, inverse=True)
        self.assertEqual(new_coords[0], 1.0)
        self.assertEqual(new_coords[1], 1.0)

    def test_apply_translation_x_center(self):
        """
        Test the proper translation of coordinates on the x-axis
        """
        self.projection.axis_to_center = 'x'
        coords = [1, 1]
        new_coords = self.projection.apply_translation(coords)
        self.assertEqual(new_coords[0], 688.3746722109003)
        self.assertEqual(new_coords[1], 1079)

    def test_apply_translation_y_center(self):
        """
        Test the proper translation of coordinates on the y-axis
        """
        self.projection.axis_to_center = 'y'
        coords = [1, 1]
        new_coords = self.projection.apply_translation(coords)
        self.assertEqual(new_coords[0], 1)
        self.assertEqual(new_coords[1], 944.0)

    @classmethod
    def tearDownClass(self):
        self.sf = None
        self.base_map = None
        self.projection = None


class TestShapeClass(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        shp_path = "geo_rendering/tests/nyc_taxi_zones/taxi_zones.shp"
        self.sf = ShapeFile(shp_path)
        self.shape = ShapeOnMap(self.sf.shapefile, 0)
        self.cv2_original = cv2.fillPoly
        cv2.fillPoly = MagicMock()

    def test_init_shape(self):
        """
        Test the proper instanciation of the Shape Class
        """
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
        """
        Test the proper computation of the coordinates attributes of a Shape instance
        """
        points = self.shape.get_shape_coords()[0]
        center = self.shape.get_shape_coords()[1]
        max_bound = self.shape.get_shape_coords()[2]
        min_bound = self.shape.get_shape_coords()[3]
        self.assertEqual(points[0], (933100.9183527103, 192536.08569720192))
        self.assertEqual(center, (935920.9674123558, 190798.94856132902))
        self.assertEqual(max_bound, (941810.0086498931, 197256.2108251324))
        self.assertEqual(min_bound, (931553.4909607167, 183788.04973023868))

    def test_project_shape_coords(self):
        """
        Test the proper conversion of the coordinates attributes of a Shape instance
        provided a projection reference
        """
        base_map = Map(self.sf, [1920, 1080])
        projection = Projection(base_map)
        self.shape.project_shape_coords(projection)
        self.assertEqual(self.shape.points[0], [555.6577658103997, 567.9118304283727])
        self.assertEqual(self.shape.center, (575.6001758964726, 580.196262509733))
        self.assertEqual(self.shape.max_bound, (617.2454355670657, 629.7749104709633))
        self.assertEqual(self.shape.min_bound, (544.7148953223824, 534.5327405728962))

    def test_fill_in_shape(self):
        """
        Test the proper modification of an image file with the filling of a given shape
        """
        map_to_render = 'Test Map'
        self.shape.fill_in_shape(map_to_render)
        # assertion statement
        cv2.fillPoly.assert_called_once()
        self.assertEqual(cv2.fillPoly.call_args[0][0], 'Test Map')
        self.assertEqual(type(cv2.fillPoly.call_args[0][1]), list)
        self.assertEqual(cv2.fillPoly.call_args[0][2], self.shape.color_fill)

    @classmethod
    def tearDownClass(self):
        self.sf = None
        self.shape = None
        cv2.fillPoly = self.cv2_original


class TestShapefileClass(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        shp_path = "geo_rendering/tests/nyc_taxi_zones/taxi_zones.shp"
        self.sf = ShapeFile(shp_path)

    def test_init_sf(self):
        """
        Test the proper instanciation of the Shapefile Class
        """
        # Test the shapefile Reader object
        self.assertEqual(len(self.sf.shapefile), 263)
        self.assertTrue(isinstance(self.sf.shapefile, shp.Reader))
        self.assertEqual(self.sf.shapefile.shapeTypeName, 'POLYGON')
        # Test the shapefile dataframe object
        self.assertEqual(self.sf.df_sf.shape, (263, 7))
        self.assertEqual(self.sf.df_sf.columns.values.tolist(), ['OBJECTID', 'Shape_Leng', 'Shape_Area', 'zone', 'LocationID', 'borough', 'coords'])
        # Test the shapefile shape_dict object
        self.assertEqual(len(self.sf.shape_dict_sf), 263)
        self.assertTrue(isinstance(self.sf.shape_dict_sf[0], ShapeOnMap))

    def test_sf_reader(self):
        """
        Test the instanciation of the Reader class from a shapefile
        """
        # Test the shapefile Reader object
        self.assertEqual(len(self.sf.shapefile), 263)
        self.assertTrue(isinstance(self.sf.shapefile, shp.Reader))
        self.assertEqual(self.sf.shapefile.shapeTypeName, 'POLYGON')

    def test_shp_to_df(self):
        """
        Test the proper construction of a dataframe from a shapefile
        """
        df_sf = self.sf.shp_to_df()
        self.assertEqual(df_sf.shape, (263, 7))
        self.assertEqual(df_sf.columns.values.tolist(), ['OBJECTID', 'Shape_Leng', 'Shape_Area', 'zone', 'LocationID', 'borough', 'coords'])

    def test_build_shape_dict(self):
        """
        Test the proper construction of a shape_dict from a dataframe
        """
        df_sf = self.sf.shp_to_df()
        shape_dict_sf = self.sf.build_shape_dict(df_sf)
        self.assertEqual(len(shape_dict_sf), 263)
        self.assertTrue(isinstance(shape_dict_sf[0], ShapeOnMap))

    def test_filter_shape_to_render_str(self):
        """
        Test the proper filtering of a dataframe when the condition statement is a string
        """
        cond_stat = 'Newark Airport'
        attr = 'zone'
        df_filtered = self.sf.filter_shape_to_render(cond_stat, attr)
        self.assertEqual(df_filtered.shape, (1, 7))
        self.assertEqual(df_filtered.iloc[0, :-1].tolist(), [1, 0.116357453189, 0.0007823067885, 'Newark Airport', 1, 'EWR'])

    def test_filter_shape_to_render_arr(self):
        """
        Test the proper filtering of a dataframe when the condition statement is an array
        """
        cond_stat = ['Newark Airport', 'Jamaica Bay']
        attr = 'zone'
        df_filtered = self.sf.filter_shape_to_render(cond_stat, attr)
        self.assertEqual(df_filtered.shape, (2, 7))
        self.assertEqual(df_filtered.iloc[0, :-1].tolist(), [1, 0.116357453189, 0.0007823067885, 'Newark Airport', 1, 'EWR'])
        self.assertEqual(df_filtered.iloc[1, :-1].tolist(), [2, 0.43346966679, 0.00486634037837, 'Jamaica Bay', 2, 'Queens'])

    def test_filter_shape_to_render_error_cond(self):
        """
        Test the proper filtering of a dataframe when the condition statement is invalid
        """
        cond_stat = 1  # wrong type of argument
        attr = 'zone'
        with self.assertRaises(AttributeError):
            self.sf.filter_shape_to_render(cond_stat, attr)

    def test_filter_shape_to_render_error_attr(self):
        """
        Test the proper filtering of a dataframe when the attribute statement is invalid
        """
        cond_stat = 'Alphabet City'
        attr = 'name'  # wrong argument value
        with self.assertRaises(KeyError):
            self.sf.filter_shape_to_render(cond_stat, attr)

    @classmethod
    def tearDownClass(self):
        self.sf = None


if __name__ == '__main__':
    unittest.main()
