from setuptools import setup, Extension
import numpy 

setup_args = dict(
    ext_modules=[Extension('amaazetools.cextensions',
                            sources=['src/cextensions.c',
                                     'src/svi_computations.c',
                                     'src/mesh_operations.c',
                                     'src/memory_allocation.c'],
                            include_dirs=[numpy.get_include()],
                            extra_compile_args = ['-Ofast'],
                            extra_link_args = ['-lm'])])

setup(**setup_args)


