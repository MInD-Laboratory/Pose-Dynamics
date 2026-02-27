"""Build script for the rqa_utils_cpp extension."""
from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        "pose_dynamics.rqa.utils.rqa_utils_cpp",
        sources=["src/pose_dynamics/rqa/utils/rqa_utils.cpp"],
        include_dirs=[
            pybind11.get_include(),
        ],
        language="c++",
        extra_compile_args=["/O2", "/std:c++17"] if __import__("sys").platform == "win32" 
                           else ["-O3", "-std=c++17"],
    ),
]

setup(
    name="rqa_utils_cpp",
    ext_modules=ext_modules,
)
