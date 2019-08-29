from setuptools import setup

setup(
    name="geo_rendering",
    version="1.0.0",
    description="Rendering of images from a shapefile using OpenCV",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/acoullandreau/geo_rendering_package",
    author="Alexina Coullandreau",
    license="MIT",
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Intended Audience :: Developers",
        "Topic :: Multimedia :: Graphics",
        "Topic :: Multimedia :: Graphics :: Viewers",
        "Topic :: Scientific/Engineering :: Visualization"
    ],
    packages=['geo_rendering', 'geo_rendering.tests'],
    install_requires=["mock==3.0.5", "numpy==1.17.0", "pandas==0.25.1", "pyshp==2.1.0", "opencv-python==4.1.0.25"],
    include_package_data=True,
    test_suite='nose.collector',
    tests_require=['nose'],
    zip_safe=False
)
