import glob
import platform

import numpy
from setuptools import Extension, setup

src_files = sorted(glob.glob("src/*.cpp"))

compile_args = [
    "-std=c++17",
    "-O3",
    "-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION",
    "-Wno-unused-function",
]
link_args = []

if platform.system() != "Darwin":
    # Linux: use OpenMP for parallelism
    compile_args.append("-fopenmp")
    link_args.append("-fopenmp")

setup(
    ext_modules=[
        Extension(
            "pyprego._pyprego",
            sources=src_files,
            include_dirs=[numpy.get_include(), "src"],
            extra_compile_args=compile_args,
            extra_link_args=link_args,
        ),
    ],
    zip_safe=False,
)
