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
    classifiers=["Programming Language :: Python :: 3",
                 "License :: OSI Approved :: MIT License",
                 "Operating System :: OS Independent"],
    python_requires='>=3.6',
)

