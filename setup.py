#This is a setup file
import setuptools
import numpy

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="amaazetools", 
    version="0.0.1",
    author="Jeff Calder",
    author_email="jwcalder@umn.edu",
    description="Python package for mesh processing tools developed by AMAAZE",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jwcalder/AMAAZETools",
    packages=['amaazetools'],
    ext_modules=[setuptools.Extension('amaazetools.cextensions',
                    sources=['src/cextensions.c',
                             'src/svi_computations.c',
                             'src/mesh_operations.c',
                             'src/memory_allocation.c'],
                    include_dirs=[numpy.get_include()],
                    extra_compile_args = ['-Ofast'],
                    extra_link_args = ['-lm'])],
    classifiers=["Programming Language :: Python :: 3",
                 "License :: OSI Approved :: MIT License",
                 "Operating System :: OS Independent"],
    install_requires=['numpy',
                      'scipy',
                      'pandas',
                      'pydicom',
                      'scikit-image',
                      'sklearn',
                      'matplotlib',
                      'graphlearning',
                      'plyfile'],
    python_requires='>=3.6',
)

