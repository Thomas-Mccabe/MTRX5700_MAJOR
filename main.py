#! /usr/bin/python3
import cv2 # state of the art computer vision algorithms library
import numpy as np # fundamental package for scientific computing
import matplotlib.pyplot as plt # 2D plotting library producing publication quality figures
import pyrealsense2 as rs # Intel RealSense cross-platform open-source API
import math as m

from handtracking.keypoints import centroid_detec
from handtracking.keypoints import findNorm
from handtracking.keypoints import plotnorm
from handtracking.keypoints import showdopthandcolor
from handtracking.keypoints import aligndepthandcolor
from handtracking.keypoints import segmentdepth
from handtracking.keypoints import centroid_detec
from handtracking.keypoints import depthto3d
from handtracking.keypoints import rotationbetweenposes

from skeletonfitting.skeletonfitting import createhand
from skeletonfitting.skeletonfitting import makeplotbox
from skeletonfitting.skeletonfitting import plotplotbox
from skeletonfitting.skeletonfitting import plothand

from basiccam.core import setupstream
from basiccam.core import getframes
# edit main
# edit main again

def main():

    LIVE = True
    SHOW_IMG = False
    file = "colordepthhand2.bag"
    #edit
    pipe, config, profile = setupstream(LIVE, file)
    plt.ion()
    X, Y, Z, ax, fig = makeplotbox(-20, 20)

    try:
        while True:

            color_frame, depth_frame, frameset = getframes(pipe)
            depth_sensor = profile.get_device().first_depth_sensor()

            aligned_color, aligned_depth = aligndepthandcolor(depth_frame,color_frame,frameset,SHOW_IMG)

            # Segment just the hand
            hand_depth = segmentdepth(aligned_depth, profile)


            blobminarea = 0;
            center_x, center_y, iamge = centroid_detec(hand_depth, blobminarea)

            #normImage = findNorm(hand_depth)
            #cv2.imshow('img', normImage)
            #cv2.waitKey(1)
            # Find the x, y, z position of the centroid of the hand.
            x, y, z = depthto3d(center_x, center_y, depth_sensor, depth_frame, aligned_depth, profile)
            #edirt
            cv2.imshow('img', iamge)
            cv2.waitKey(1)
            #print(' ')

            angg = 30*m.pi/180

            pinkf = [angg,angg,angg]
            ringf = [0,0,0]
            middf = [0,0,0]
            indxf = [0,0,0]
            thumb = [0,0,0]
            fingers_angle = [pinkf,ringf,middf,indxf,thumb]

            pos = [x/100,y/100,z/100]
            ax.clear()
            ax.view_init(elev = 70, azim = 30)
            # ax.scatter(x/100,y/100,z/100, label='True Position',color='blue')

            # These angles assume that there is a right hand facing the camera
            # & That the fingers face up

            n, b, c = findNorm(aligned_depth,depth_frame, depth_sensor, profile, center_x,center_y)
            plotnorm(n, b, c)

            pose1 = [[1,0,0],[0,1,0],[0,0,1]]
            pose2 = [[n],[b],[c]]
            rot = rotationbetweenposes(pose1, pose2)


            hand_measure = createhand(fingers_angle,pos,rot)
            
            # print(hand_measure)
            plothand(hand_measure, ax, 'red')
            plotplotbox(X, Y, Z, ax, fig)

            plt.show()
            plt.pause(0.01)




            # Now to detect the fingers and plot them

            #input()


            # Could assume if the height and width of the hand are within
            # Some range then they are closed or open
            # This will correspond to the skeleton in unity that will move

    finally:
        pipe.stop()

if __name__ == '__main__':
    # do main stuff
    main()
