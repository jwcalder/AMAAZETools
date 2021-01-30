#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 13:04:55 2021

@author: rileywilde
"""

from distutils.core import setup, Extension
import numpy

# define the extension module
svipca_module = Extension('svipca_module', sources=['svipca_py.c','svi_computations.c','mesh_operations.c','memory_allocation.c'],include_dirs=[numpy.get_include()],extra_compile_args = ['-Ofast'])

# run the setup
setup(ext_modules=[svipca_module])
