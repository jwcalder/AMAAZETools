from distutils.core import setup, Extension
import numpy

# define the extension module
svi_module = Extension('svi_module', sources=['svi_py.c','svi_computations.c','mesh_operations.c','memory_allocation.c'],include_dirs=[numpy.get_include()],extra_compile_args = ['-Ofast'])

# run the setup
setup(ext_modules=[svi_module])
