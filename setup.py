from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import sysconfig
import os
import subprocess

try:
    import pybind11
except ImportError:
    print("pybind11 is not installed. Please install it first: pip install pybind11")
    sys.exit(1)


# Helper to get OpenCL include and lib dirs if needed
def get_opencl_flags():
    """Return OpenCL compiler and linker flags."""
    if os.name == 'nt':
        # Windows
        # Assuming OpenCL.lib and CL/cl.h are in standard paths (e.g., from AMD SDK)
        # You may need to customize this if headers or libs are in custom locations
        return {
            'include_dirs': [],
            'libraries': ['OpenCL'],
            'library_dirs': [],
            'extra_compile_args': [],
        }
    else:
        # Linux / WSL / Mac
        return {
            'include_dirs': [],
            'libraries': ['OpenCL'],
            'library_dirs': [],
            'extra_compile_args': ['-fPIC'],
        }


opencl_flags = get_opencl_flags()

ext_modules = [
    Extension(
        name='clDPM',  # This will produce clDPM[.pyd/.so]
        sources=[
            os.path.join("src", f)
            for f in os.listdir("src")
            if f.endswith(".cpp")
        ],
        include_dirs=[
            pybind11.get_include(),
            sysconfig.get_paths()["include"],
            os.path.join(os.getcwd(), "include"),
        ] + opencl_flags['include_dirs'],
        libraries=opencl_flags['libraries'],
        library_dirs=opencl_flags['library_dirs'],
        language='c++'
    )
]


class BuildExt(build_ext):
    """Custom build_ext to print compiler flags"""
    def build_extensions(self):
        ct = self.compiler.compiler_type
        print(f"Compiler type: {ct}")
        for ext in self.extensions:
            print(f"Building extension: {ext.name}")
            print(f"  Sources: {ext.sources}")
            print(f"  Include dirs: {ext.include_dirs}")
            print(f"  Libraries: {ext.libraries}")
            print(f"  Extra compile args: {ext.extra_compile_args}")
        build_ext.build_extensions(self)


setup(
    name='clDPM',
    version='1.0',
    author='Shaka',
    description='C++ OpenCL GPU Accelerated DPModel',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExt},
    zip_safe=False,
)
