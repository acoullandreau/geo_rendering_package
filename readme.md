geo_rendering package - documentation
=========================================

a folder with the name of your package that contains:
the Python code that makes up your package
a README.md file
an __init__.py
license.txt
setup.cfg
setup.py file



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

	$ pip install geo-rendering $ 

It relies in particular on two libraries: OpenCV and pyshp.




Tutorial
---------

A brief introduction on how to use this package is available here:


Code documentation
------------------


Package functions
-----------------


Test functions
-----------------

There is a total of 37 tests available, that should cover most of the classes and methods available. 


| Name 		   | Stmts | Miss | Cover | Missing         |
| ------------ | ----- | ---- | ----- | --------------- |
| classfile.py | 214   |  10  |  95%  | 20-28, 155-156  |
| utility.py   | 33    |   0  |  100% |                 |


Further work and improvements
-----------------------------

Some improvements that could be performed on the package:

- convert id should not go beyond 0
- handle errors if points in calculate_boundaries are not lists of tuples
- handle errors if points in calculate_centroid are not lists of tuples
- handle errors if points to render on map is not a list of coordinates (type POLYGON)
- format of input for interpolate_next_position
- in general, handle better errors related to the format of an input



Sources, acknowlegments and related content
-------------------------------------------



