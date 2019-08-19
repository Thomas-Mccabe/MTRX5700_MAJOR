#! /usr/bin/python3
import cv2 # state of the art computer vision algorithms library
import numpy as np # fundamental package for scientific computing
import matplotlib.pyplot as plt # 2D plotting library producing publication quality figures
import pyrealsense2 as rs # Intel RealSense cross-platform open-source API
import math as m
import time

from handtracking.keypoints import centroid_detec
from handtracking.keypoints import findNorm
from handtracking.keypoints import showdopthandcolor
from handtracking.keypoints import aligndepthandcolor
from handtracking.keypoints import segmentdepth
from handtracking.keypoints import centroid_detec
from handtracking.keypoints import depthto3d

from basiccam.core import setupstream
from basiccam.core import getframes

from skeletonfitting.skeletonfitting import animatehand
from skeletonfitting.skeletonfitting import createhand
from skeletonfitting.skeletonfitting import makeplotbox
from skeletonfitting.skeletonfitting import plotplotbox
from skeletonfitting.skeletonfitting import fingersfromhand
from skeletonfitting.skeletonfitting import plothand
from skeletonfitting.skeletonfitting import distbetweenpoints
from skeletonfitting.skeletonfitting import findhand

# Create some artificial measurement
pinkf = [30,30,30]
ringf = [90,10,0]
middf = [0,0,0]
indxf = [10,10,10]
thumb = [40,80,30]
fingers_angle = [pinkf,ringf,middf,indxf,thumb]

pos = [0.1,-0.05,0]
rot = [0,0.001,0]
hand_measure = createhand(fingers_angle,pos,rot)
measHand = fingersfromhand(hand_measure)

meas_pos = [0,0,0]
meas_rot = [0,0,0]

sim_hand = findhand(measHand, meas_pos, meas_rot)

X, Y, Z, ax, fig = makeplotbox(-10, 30)
plotplotbox(X, Y, Z, ax, fig)
plothand(sim_hand, ax, 'blue')
plothand(hand_measure, ax, 'red')

plt.show()
plt.pause(0.05)

input()
ax.clear()

#

# # For testing
# #animatehand()
# #  Create some hand with known orientation + some noise
# #  and known position + some noise
# #  But unknown fingertip positions
# pos = [10,10,0]
# rot = [0,0,0]
#
#
# pinkf = [30,30,30]
# ringf = [90,10,0]
# middf = [0,0,0]
# indxf = [10,10,10]
# thumb = [40,80,30]
# fingers_angle = [pinkf,ringf,middf,indxf,thumb]
# hand_measure = createhand(fingers_angle,pos,rot)
# measHand = fingersfromhand(hand_measure)
#
#
# pinkf = [0,0,0]
# ringf = [0,0,0]
# middf = [0,0,0]
# indxf = [0,0,0]
# thumb = [0,0,0]
# sim_angle = [pinkf,ringf,middf,indxf,thumb]
# hand_sim = createhand(sim_angle,pos,rot)
# simHand = fingersfromhand(hand_sim)
#
# # There should be some association between the sim and measured
# # finger positions where each distance to each is measured and the
# # minimum difference is saved.
#
# dist = np.zeros((5,5))
# print(measHand[1][0])
# for i in range(0,dist.shape[0]):
#     for j in range(0,dist.shape[0]):
#         dist[i][j] = m.sqrt((measHand[i][0][0] - simHand[j][0][0])**2 + (measHand[i][0][1] - simHand[j][0][1])**2 + (measHand[i][0][2] - simHand[j][0][2])**2)
#
# # This says which measured finger point is the corresponding sim finger point to minimise
# # Find the minimum row in the dist matrix & this will associate the fingers.
#
# Pme_to_sim = dist[0].argmin()
# Rme_to_sim = dist[1].argmin()
# Mme_to_sim = dist[2].argmin()
# Ime_to_sim = dist[3].argmin()
# Tme_to_sim = dist[4].argmin()
#
#
# # Then run a loop moving the sim hand to reduce the total sum of the squares of all the fingers
# # for a bunch of random finger poses save the pose that minimises the square of the error
# j1_angles = [0*m.pi/180, 22*m.pi/180, 44*m.pi/180, 66*m.pi/180, 90*m.pi/180]
# j2_angles = [0*m.pi/180, 22*m.pi/180, 44*m.pi/180, 66*m.pi/180, 90*m.pi/180]
# j3_angles = [0*m.pi/180, 10*m.pi/180, 20*m.pi/180, 30*m.pi/180]
#
# fingers = [0,1,2,3,4]
#
# X, Y, Z, ax, fig = makeplotbox(-10, 10)
# start = time.clock()
#
# # Set the initial minimum idstances to be inf
# Pmin = float('inf')
# Rmin = float('inf')
# Mmin = float('inf')
# Imin = float('inf')
# Tmin = float('inf')
#
# Pmin_state = [0,0,0]
# Rmin_state = [0,0,0]
# Mmin_state = [0,0,0]
# Imin_state = [0,0,0]
# Tmin_state = [0,0,0]
#
# for j1 in j1_angles:
#     for j2 in j2_angles:
#         for j3 in j3_angles:
#             pinkf = [j1,j2,j3]
#             ringf = [j1,j2,j3]
#             middf = [j1,j2,j3]
#             indxf = [j1,j2,j3]
#             thumb = [j1,j2,j3]
#
#             sim_angle = [pinkf,ringf,middf,indxf,thumb]
#             HH = createhand(sim_angle,pos,rot)
#
#             # Measures the error between the real pinky and the sim pinky Pme_to_sim
#             simHand2 = fingersfromhand(HH)
#             Perror = distbetweenpoints(measHand, simHand2, 0, Pme_to_sim)
#             Rerror = distbetweenpoints(measHand, simHand2, 1, Rme_to_sim)
#             Merror = distbetweenpoints(measHand, simHand2, 2, Mme_to_sim)
#             Ierror = distbetweenpoints(measHand, simHand2, 3, Ime_to_sim)
#             Terror = distbetweenpoints(measHand, simHand2, 4, Tme_to_sim)
#
#
#             # Now to save the state of the minimum distance
#             if Perror < Pmin:
#                 Pmin = Perror
#                 Pmin_state = [j1, j2, j3]
#             if Rerror < Rmin:
#                 Rmin = Rerror
#                 Rmin_state = [j1, j2, j3]
#             if Merror < Mmin:
#                 Mmin = Merror
#                 Mmin_state = [j1, j2, j3]
#             if Ierror < Imin:
#                 Imin = Ierror
#                 Imin_state = [j1, j2, j3]
#             if Terror < Tmin:
#                 Tmin = Terror
#                 Tmin_state = [j1, j2, j3]
#
#
# end = time.clock()
# print('Hand track time')
# print(end-start)
#
#
# sim_angle = [Pmin_state,Rmin_state,Mmin_state,Imin_state,Tmin_state]
# HH = createhand(sim_angle,[5,5,0],rot)
#
# return HH
#
#
# X, Y, Z, ax, fig = makeplotbox(-10, 30)
# plotplotbox(X, Y, Z, ax, fig)
# plothand(HH, ax, 'blue')
# plothand(hand_measure, ax, 'red')
#
# input()
# ax.clear()
#
# # look into sim noisy hand
# # look into usingonly one vector the palm normal
# # look into seeing if hands should fold back or forward based on if points are above or below surface norm
# # Maybe it will be a case of if the fingers are firther back than the palm (obvious issues there)
