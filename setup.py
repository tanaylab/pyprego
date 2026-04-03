import glob

import numpy
from setuptools import Extension, setup

src_files = sorted(glob.glob("src/*.cpp"))

setup(
    ext_modules=[
        Extension(
            "pyprego._pyprego",
            sources=src_files,
            include_dirs=[numpy.get_include(), "src"],
            extra_compile_args=[
                "-std=c++17",
                "-O3",
                "-fopenmp",
                "-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION",
                "-Wno-unused-function",
            ],
            extra_link_args=["-fopenmp"],
        ),
    ],
    zip_safe=False,
)
