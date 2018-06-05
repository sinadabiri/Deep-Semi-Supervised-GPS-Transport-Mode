import matplotlib.pyplot as plt
import pickle
import time
import gmplot
import random
import numpy as np
# Creating heat map for extracted pairs of lat and long
a = [[1, 2], [2, 2], [3, 3], [3, 4], [4, 5]]

b = filter(lambda x: x[1], a)
filename = '../Mode-codes-Revised/paper2_Trajectory_Label.pickle'
with open(filename, 'rb') as f:
    trajectory_all_user_with_label, trajectory_all_user_wo_label = pickle.load(f)
lat = []
long = []
Beijing = [(39.438178, 41.079483), (115.420143, 117.522504)]  # Beijing GPS coord boundaries.
for user in trajectory_all_user_with_label:
    for gps in user[:1200]:
        if Beijing[0][0] <= gps[0] <= Beijing[0][1] and Beijing[1][0] <= gps[1] <= Beijing[1][1]:
            lat.append(gps[0])
            long.append(gps[1])

for user in trajectory_all_user_wo_label:
    for gps in user[:7000]:  # Using only 7000 GPS logs of every user to make HTML file smaller.
        if Beijing[0][0] <= gps[0] <= Beijing[0][1] and Beijing[1][0] <= gps[1] <= Beijing[1][1]:
            lat.append(gps[0])
            long.append(gps[1])
# Make half the data point to visualize
lat = lat[:int(1*len(lat))]
long = long[:int(1*len(long))]
gmap = gmplot.GoogleMapPlotter(39.972353, 116.331708, 12)
gmap.heatmap(lat, long)
gmap.draw('HeatMap-Distribution of GPS trajectories in Beijing.html')