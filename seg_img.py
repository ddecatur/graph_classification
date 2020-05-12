import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import colorsys
from k_means_clustering import *


# type: img path --> list of color ranges
def idColor(image):#'./testttttt.png'):
    rangeList = list()
    
    # convert image
    #image = cv2.imread(image)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # convert color
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    
    
    clt = KMeansCluster(image)
    hist = clusterCounts(clt)

    # determine which colors to segment out
    colList = list()
    maxCol = max(hist) # identify the background color as the most common color so it can be removed
    minCol = min(hist)
    for (percent, color) in zip(hist, clt.cluster_centers_):
        if (percent != maxCol and percent != minCol):
            colList.append(color.astype("uint8").tolist())
    return colList

def rgb_hsv(r,g,b): # code adapted from: https://www.geeksforgeeks.org/program-change-rgb-color-model-hsv-color-model/ to work for openCV representation of hsv
    # R, G, B values are divided by 255 
    # to change the range from 0..255 to 0..1: 
    r, g, b = r / 255.0, g / 255.0, b / 255.0
  
    # h, s, v = hue, saturation, value 
    cmax = max(r, g, b)    # maximum of r, g, b 
    cmin = min(r, g, b)    # minimum of r, g, b 
    diff = cmax-cmin       # diff of cmax and cmin. 
  
    # if cmax and cmax are equal then h = 0 
    if cmax == cmin:  
        h = 0
      
    # if cmax equal r then compute h 
    elif cmax == r:  
        h = (60 * ((g - b) / diff) + 360) % 360
        h = h/2
  
    # if cmax equal g then compute h 
    elif cmax == g: 
        h = (60 * ((b - r) / diff) + 120) % 360
        h = h/2
  
    # if cmax equal b then compute h 
    elif cmax == b: 
        h = (60 * ((r - g) / diff) + 240) % 360
        h =h/2
  
    # if cmax equal zero 
    if cmax == 0: 
        s = 0
    else: 
        s = (diff / cmax) * 100
    s = (s/100)*255
  
    # compute v 
    v = cmax * 100
    v = (v/100)*255
    return (h,s,v)

# given a list of rgb colors return a list of hsv ranges
def hsvRange (rgbList):
    rangeList = list()
    for (r,g,b) in rgbList:
        (h,s,v) = rgb_hsv(r,g,b)
        ll = (max(h-15,0), max(s-15,0), max(v-40,0))
        ul = (min(h+15,180), min(s+15,255), min(v+40,255))
        rangeList.append((ll,ul))
    print(rangeList)
    return rangeList


def segmentImg(img='./graphs_filtered/testttttt.png'):
    '''
    Givn an image path, segment out any blue from that image
    '''
    # read in image
    graph = cv2.imread(img)
    graph = cv2.cvtColor(graph, cv2.COLOR_BGR2RGB) # convert color from BGR to RGB
    colRangeList = hsvRange(idColor(graph))

    # color range for the mask -- (0, 0, 30), (80, 80, 255)
    #lower_blue = np.array([110,50,50])
    #upper_blue = np.array([130,255,255])
    #lower_red = np.array([0,0,30])
    #upper_red = np.array([80,80,255])
    #lower_green = np.array([40,40,40])
    #upper_green = np.array([70,255,255])



    # convert to hsv image type
    hsv_graph = cv2.cvtColor(graph, cv2.COLOR_RGB2HSV)
    results = list()
    for (ll, ul) in colRangeList:
        mask = cv2.inRange(hsv_graph, ll, ul)
        results.append(cv2.bitwise_and(graph, graph, mask=mask))

    #maskBlue = cv2.inRange(hsv_graph, lower_blue, upper_blue)
    #maskRed = cv2.inRange(hsv_graph, lower_red, upper_red)
    #maskGreen = cv2.inRange(hsv_graph, lower_green, upper_green)
    #maskF = maskBlue + maskRed + maskGreen # use all three main color masks
    #result = cv2.bitwise_and(graph, graph, mask=mask)

    # show image
    #plt.subplot(1, len(results)+1, 1)
    #plt.imshow(mask, cmap="gray")
    #i=2
    #for res in results:
    #    plt.subplot(1, len(results)+1, i)
    #    plt.imshow(res)
    #    i=i+1
    #plt.show()

    return results


# Function to save the graphs
def saveGraphs(rootDir='graphs_filtered'):
  
    for count, dir_name1 in enumerate(os.listdir(rootDir)): # options: train, validation
        if dir_name1 != '.DS_Store':
            for count, dir_name2 in enumerate(os.listdir(rootDir+'/'+dir_name1)): # options: negative, neutral, positive
                if dir_name2 != '.DS_Store':
                    for count, filename in enumerate(os.listdir(rootDir+'/'+dir_name1+'/'+dir_name2)): # options: actual image names
                        segImg = segmentImg(rootDir+'/'+dir_name1+'/'+dir_name2+'/'+filename)
                        i=0
                        for res in segImg:
                            fname = rootDir+'/'+dir_name1+'/'+dir_name2+'/seg_'+str(i)+'_'+filename
                            plt.imsave(fname, res)
                            i=i+1

def tests():
    img = './graphs_filtered/train/negative/reg_bar_graph3.png'
    graph = cv2.imread(img)
    graph = cv2.cvtColor(graph, cv2.COLOR_BGR2RGB)
    print(hsvRange(idColor(graph)))
#saveGraphs()
#tests()
#segmentImg()