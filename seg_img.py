import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image


def segmentImg(img='./testttttt.png'):
    '''
    Givn an image path, segment out any blue from that image
    '''
    # read in image
    graph = cv2.imread(img)
    graph = cv2.cvtColor(graph, cv2.COLOR_BGR2RGB) # convert color

    # color range for the mask -- (0, 0, 30), (80, 80, 255)
    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])
    lower_red = np.array([0,0,30])
    upper_red = np.array([80,80,255])
    lower_green = np.array([40,40,40])
    upper_green = np.array([70,255,255])



    # convert to hsv image type
    hsv_graph = cv2.cvtColor(graph, cv2.COLOR_RGB2HSV)

    maskBlue = cv2.inRange(hsv_graph, lower_blue, upper_blue)
    maskRed = cv2.inRange(hsv_graph, lower_red, upper_red)
    maskGreen = cv2.inRange(hsv_graph, lower_green, upper_green)
    maskF = maskBlue + maskRed + maskGreen # use all three main color masks
    result = cv2.bitwise_and(graph, graph, mask=maskBlue)

    # show image
    #plt.subplot(1, 2, 1)
    #plt.imshow(maskF, cmap="gray")
    #plt.subplot(1, 2, 2)
    #plt.imshow(result)
    #plt.show()

    return result


# Function to save the graphs
def saveGraphs(rootDir='graphs_filtered'):
  
    for count, dir_name1 in enumerate(os.listdir(rootDir)): # options: train, validation
        if dir_name1 != '.DS_Store':
            for count, dir_name2 in enumerate(os.listdir(rootDir+'/'+dir_name1)): # options: negative, neutral, positive
                if dir_name2 != '.DS_Store':
                    for count, filename in enumerate(os.listdir(rootDir+'/'+dir_name1+'/'+dir_name2)): # options: actual image names
                        segImg = segmentImg(rootDir+'/'+dir_name1+'/'+dir_name2+'/'+filename)
                        fname = rootDir+'/'+dir_name1+'/'+dir_name2+'/seg_'+filename
                        plt.imsave(fname, segImg)