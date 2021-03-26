# -*- coding: utf-8 -*-
import csv
import glob
#import os
import amaazetools.trimesh as tm

# This script loads all the ply files in a directory and outputs
# a new csv file that contains the relevant data from the mesh

# If you need to change directories:
#directory = ""
#os.chdir(directory)

# This grabs all the files with the .ply in the current directory
mesh_list = []
for file in glob.glob("*.ply"):
    mesh_list.append(file)

# Header order:
header = ["Label", "Surface Area", "Volume", "BB1", "BB2", "BB3"] # This is the order of variables

# Reading the ply files into data
data_list = []
for label in mesh_list:
    ply = tm.load_ply(label)
    surf_area = ply.surf_area()
    volume = ply.volume()
    bb = ply.bbox() # The bounding box variables are assumed to be in the same order each time
    ls = [label, surf_area, volume, bb[0], bb[1], bb[2]]  # This order can be changed, but you gotta change the header too
    data_list.append(ls)

# Outputing the data into a csv
file_out = "out.csv"
with open(file_out, mode = 'w', newline='') as data_file:
    data_writer = csv.writer(data_file)
    data_writer.writerow(header) # Write Header
    data_writer.writerows(data_list) # This writes all the data in the data list in the order it's stored in