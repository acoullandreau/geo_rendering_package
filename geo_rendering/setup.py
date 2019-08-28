import pathlib
from setuptools import setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "readme.md").read_text()

# This call to setup() does all the work
setup(
    name="geo_rendering",
    version="1.0.0",
    description="Rendering of images from a shapefile using OpenCV",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/acoullandreau/geo_rendering_package",
    author="Alexina Coullandreau",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    packages=["geo_rendering"],
    include_package_data=True,
    install_requires=["mock", "numpy", "pandas", "pyshp", "opencv-python"],
    test_suite='nose.collector',
    tests_require=['nose']
)
