# -*- coding: utf-8 -*-
import csv
import amaazetools.trimesh as tm

# This script loads all the ply files named in a csv file and outputs
# a new csv file that contains the relevant data from the mesh

# Reading from the csv for the files to process
file_in = "exampleInput.csv"
mesh_list = []
with open(file_in, mode = 'r') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter = ',')
    header_flag = True # If the input csv file doesn't have a header line, set this to false
    for row in csv_reader:
        if header_flag == True:
            header_flag = False
            continue
        mesh_list.append(row[0])

# Header order:
header = ["Label", "Surface Area", "Volume", "BB1", "BB2", "BB3"] # This is the order of variables

# Reading the ply files into data
data_list = []
for label in mesh_list:
    ply = tm.load_ply(label + ".ply") # The name in the input csv file is assumed to not contain the ply extension
    surf_area = ply.surf_area()
    volume = ply.volume()
    bb = ply.bbox() # The bounding box variables are assumed to be in the same order each time
    ls = [label, surf_area, volume, bb[0], bb[1], bb[2]]  # This order can be changed, but you gotta change the header too
    data_list.append(ls)

# Outputing the data into a csv
file_out = "out.csv"
with open(file_out, mode = 'w', newline='') as data_file:
    data_writer = csv.writer(data_file)
    data_writer.writerow(header) # First row is header
    data_writer.writerows(data_list) # This writes all the data in the data list in the order it's stored in