geo_rendering package - documentation
=========================================


Purpose of this package
-----------------------

This package is designed to render maps from a **shapefile**.
It allows you to:

- create an image file of any size with the shapefile map rendered on it
- convert the projection system to scale the shapefile map on the image_size
- draw points and fill shapes on the map
- add text on the map


Requirements
------------

The code is written in Python 3.7.

There are several dependencies for this package. The versions provided here are those used for testing.

- numpy 1.16.4
- opencv-python 4.1.0
- pandas 0.25.0
- pyshp 2.1.0
- unittest (python standard library)
- mock 3.0.5 (or unittest.mock)


Installation
------------

To install the package:

```
pip install geo_rendering
```

It relies in particular on two libraries: OpenCV and pyshp.
The setup.py file lists all dependencies.

Tutorial
---------

A brief introduction on how to use this package is available on [Medium](https://medium.com/@mozart38).


Code documentation
------------------

Package functions
-----------------

**ContextualText** is the class responsible for text rendering. 
An instance is created provided three arguments:
- content: the text to display
- position: the coordinates (in pixels) of the bottom left point of the text - 
- color: as a tuple of BGR color

Other attributes are set by default but can be edited at any time:
- font (default cv2.FONT_HERSHEY_SIMPLEX)
- font_style (default cv2.LINE_AA)
- text_size (default 1)
- thickness (default 1)

Only one method is available: 
- display_text(map_to_edit), that renders the text on the map file provided as an argument

See [OpenCV ](https://docs.opencv.org/3.0-beta/modules/imgproc/doc/drawing_functions.html) requirements for cv2.putText for more details. 

-----------------

**Map** is the class responsible for the rendering of a map provided a shapefile. 

An instance is created provided three arguments:
- shapefile: a Shapefile class instance
- image_size: the image size (width, height) in pixels
- background_color[optional]: the color of the background image (default black)

Other attributes are set by default but can be edited at any time:
- shape_dict: a dictionary of all shapes in the shapefile (key is the index of the shape, value a instance of the Shape class)
- shape_dict_filt: equivalent to the shape_dict unless part of the shapefile is to be rendered (filter)
- max_bound / min_bound: the extreme coordinates of the boundaries of the shapefile
- projection (default {})
- map_file: the image file

Three methods are available: 
- build_shape_dict(ref_df): constructs the dictionary using the id of each shape from the shapefile and instantiating instances of the Shape class for each of them ; it uses the dataframe extracted from the shapefile (see below)
- find_max_coords(): computes the max_bound / min_bound
- render_map(): generates an image file with a size of image_size and plots each shape of the shape_dict_filt

See [OpenCV ](https://docs.opencv.org/3.0-beta/modules/imgproc/doc/drawing_functions.html) requirements for cv2.popylines for more details. 

-----------------

**PointOnMap** is the class responsible for point rendering. 

An instance is created provided three arguments:
- coordinates: the coordinates (in pixels) of the center of the point
- weight: the size of the point to render
- color: as a tuple of BGR color

Other attributes are set by default but can be edited at any time:
- origin coordinates (x_coord_or/y_coord_or)
- current coordinates (x_coord_curr/y_coord_curr): by default the same as the origin coordinates, unless the point should have an position interpolated using the dedicated method

Two methods are available: 
- render_point_on_map(base_map), that renders the point on the map file provided as an argument
- interpolate_next_position(target_coords, tot_frames, curr_frame), used to interpolate the position of a point, knowing where we want it to arrive, in how many 'hops' ; especially used when rendering multiple frames for an animation

See [OpenCV ](https://docs.opencv.org/3.0-beta/modules/imgproc/doc/drawing_functions.html) requirements for cv2.circle for more details. 

-----------------

**Projection** is the class responsible for calculating the conversion ratio and axis to use to center the map, from the coordinate system of the shapefile to the new coordinate system of the image file.

An instance is created provided two arguments:
- map_to_scale: the Map class instance object that we want to render
- margins [optional]: margins to add to (top, right, bottom, left) of the image file in pixels


Other attributes are set by default but can be edited at any time:
- image_size, an attribute of the map_to_scale
- map_max_bound / map_min_bound: an attribute of the map_to_scale
- conversion: calculated using define_projection method
- axis_to_center: calculated using define_projection method

Three methods are available: 
- define_projection(): calculates the conversion rate and axis on which to center the converted coordinates
- apply_projection(coords, inverse=False): applies the conversion on coordinates ; the inverse argument allows to go from one coordinate system (the original one), to the new one
- apply_translation(coords): translates the coordinates along the axis to center in order to center the map


-----------------

**ShapeFile** is the class responsible for converting a shapefile into a reader object and a dataframe.

An instance is created provided one single argument:
- shp_path, the path of the shapefile folder ; note that this folder should contain at least a .shp, .dbf and .shx in the same folder

Other attributes are set by default but can be edited at any time:
- shapefile, the shapefile reader object
- df_sf the information of the shapefile in the form of a dataframe
- shape_dict_sf: a dictionary of all shapes in the shapefile (key is the index of the shape, value a instance of the Shape class)

Four methods are available: 
- sf_reader(path): creates a reader object from the shapefile folder
- shp_to_df(): converts the shapefile reader to a dataframe
- filter_shape_to_render(cond_stat, attr): filters out the dataframe using a particular column (attr) and a condition statement (cond_stat) ; cond_stat can be either a single string or a list of strings to match in the provide attr column
- build_shape_dict(ref_df): constructs the dictionary using the id of each shape from the shapefile and instantiating instances of the Shape class for each of them ; it uses the dataframe extracted from the shapefile

See [PySHP](https://pypi.org/project/pyshp/) requirements for more details. 


-----------------

**ShapeOnMap** is the class responsible for shape rendering. 

An instance is created provided three arguments:
- shapefile: a Shapefile class instance
- shape_id: the id of the shape (can be assigned arbitrarily, but it is intended to be the same that the one used in the shapefile dataframe)

Other attributes are set by default but can be edited at any time:
- points: the coordinates of the boundaries of the shape
- center: the center coordinates of the shape
- max_bound / min_bound: the extreme coordinates of the boundaries of the shape
- color_line (default white)
- line_thick (default 1)
- color_fill (default black)

Three methods are available: 
- get_shape_coords(): calculates the points, center, max_bound / min_bound coordinates using the shapefile reader 'shape' attribute
- project_shape_coords(projection): converts the coordinates of the shape using a given projection (conversion, axis_to_center), relying on the two projection methods apply_projection() and apply_translation()
- fill_in_shape(map_to_render): renders the shape on the map file provided as an argument


See [OpenCV ](https://docs.opencv.org/3.0-beta/modules/imgproc/doc/drawing_functions.html) requirements for cv2.fillPolyfor more details. 

-----------------

**Utils** is a class containing static methods used in other classes.

The three methods are:
- calculate_centroid(points): computes the value of the center of a list of points
- calculate_boundaries(points): returns the value of the max and min coordinates of the boundary delimited by the points
- convert_id(idx, inverse=False): converts an index from one range [1-x] to another [0-x], or the inverse (if inverse=True)



Test functions
-----------------

There is a total of 37 tests available, that should cover most of the classes and methods available. 


| Name 		   | Stmts | Miss | Cover | Missing         |
| ------------ | ----- | ---- | ----- | --------------- |
| classfile.py | 214   |  10  |  95%  | 20-28, 155-156  |
| utility.py   | 33    |   0  |  100% |                 |

Note: tests can be executed using the following command from the geo_rendering main directory (where the setup.py file is stored!)

```
python setup.py test
```


Further work and improvements
-----------------------------

Some improvements that could be performed on the package:

- convert id should not go beyond 0
- handle errors if points in calculate_boundaries are not lists of tuples (it doesn't make sense otherwise)
- handle errors if points in calculate_centroid are not lists of tuples (it doesn't make sense otherwise)
- handle errors if points to render on map is not a list of coordinates (type POLYGON)
- format of input for interpolate_next_position
- in general, handle better errors related to the format of an input


Sources, acknowlegments and related content
-------------------------------------------

This work is inspired from a data visualisation project about the NYC taxi rides ([link to the repository of the project](https://github.com/acoullandreau/nyc_taxi_trips))

