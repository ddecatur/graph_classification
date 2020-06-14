import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as pltcol
import numpy as np
import os
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import colorsys
from k_means_clustering import *
import math

def col_dist(c1,c2):
    (r,g,b) = c1
    (R,G,B) = c2
    return math.sqrt((r-R)**2 + (g-G)**2 + (b-B)**2)

def find_nearest_col_rgb(color, dic):
    dist = list()
    (R,G,B) = color
    minD = 442 # max color distance
    minC = tuple() # tuple
    for elem in dic:
        d = col_dist(elem,color) #math.sqrt((r-R)^2 + (g-G)^2 + (b-B)^2)
        if d < minD:
            minD = d
            minC = elem#dic.get(elem)
    return minC

def isGrayish(color):
    (r,g,b) = color
    ac = (r+g+b)/3
    cd = col_dist((ac,ac,ac), color)
    if cd < 20:
        return True
    else:
        return False

# type: img path --> list of color ranges
def idColor(image):#'./testttttt.png'):
    rangeList = list()
    
    ogImg = image
    # convert image
    image = cv2.imread(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # convert color
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    

    clt = KMeansCluster(image,ogImg)
    hist = clusterCounts(clt)
    posRGB = {(255,0,0):'r', (0,128,0):'g', (0,0,255):'b', (0,0,0):'black', (255,255,255):'white'}
    # determine which colors to segment out
    colList = list()
    maxCol = max(hist) # identify the background color as the most common color so it can be removed
    minCol = min(hist)
    #print(pltcol.get_named_colors_mapping())
    for (percent, color) in zip(hist, clt.cluster_centers_):
        nearcol=find_nearest_col_rgb(color,posRGB)
        # if nearcol==(255,0,0):
        #print('percent,near,color,max: ')
        #print(percent, nearcol, color, maxCol)
        if (percent != maxCol and isGrayish(color)==False): #percent != minCol):#nearcol!=(0,0,0)) and nearcol!=(255,255,255):# and nearcol!='white'):# and col_dist(color,(0,0,0))>60):# and percent > 0.02): # and percent != minCol):
            #colList.append(nearcol)
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

def find_ranges(h,s,v):
    rtn = set()
    lh = list()
    uh = list()
    low = h-10
    high = h+10
    lh.append(max(low,0))
    uh.append(min(high,180))
    lspill = 0-low
    uspill = high-180
    if lspill > 0:
        lh.append(180-lspill)
        uh.append(180)
    if uspill > 0:
        lh.append(0)
        uh.append(uspill)
    sl = max(s-10,0)
    vl = max(v-40,0) # should be -40
    su = min(s+10,255) # should be +10
    vu = min(v+40,255)
    for i,bound in enumerate(lh):
        rtn.add(((bound, sl, vl), (uh[i], su, vu)))
    return rtn
    
    

# given a list of rgb colors return a list of hsv ranges
def hsvRange (rgbList):
    rangeList = list()
    for (r,g,b) in rgbList:
        (h,s,v) = rgb_hsv(r,g,b)
        # (hl,hu) = wrap_around_h(h)
        # ll = (max(h-20,0), max(s-40,0), max(v-40,0))
        # ul = (min(h+20,180), min(s+40,255), min(v+40,255))
        # RS.add((ll,ul))
        # if 
        rangeList.append(find_ranges(h,s,v))
        
    # total = zip(rangeList, rgbList)
    # for item in total:
    #     print (item)
    return rangeList

def hsv_col_str():
    return 2

def segmentImg(img='./testttttt.png'):
    '''
    Givn an image path, segment out any blue from that image
    '''
    # read in image
    graph = cv2.imread(img)
    graph = cv2.cvtColor(graph, cv2.COLOR_BGR2RGB) # convert color from BGR to RGB
    colList = idColor(img)
    colRangeList = hsvRange(colList)

    # convert to hsv image type
    hsv_graph = cv2.cvtColor(graph, cv2.COLOR_RGB2HSV)
    #print(hsv_graph)
    results = list()
    for i,rngSet in enumerate(colRangeList):
        masks = list()
        for (ll,ul) in rngSet:
            masks.append(cv2.inRange(hsv_graph, ll, ul))
        #mask = cv2.inRange(hsv_graph, ll, ul)
        maskFinal = masks[0]
        # for mask in masks:
        #     if maskFinal == None:
        #         maskFinal = mask
        #     else:
        #         maskFinal = maskFinal + mask
        for j in range(1,len(masks)):
            maskFinal = maskFinal + masks[j]
        results.append((cv2.bitwise_and(graph, graph, mask=maskFinal), colList[i]))

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
                        # col_corr = 23#det_corr(filename)
                        # for (res,col) in segImg:
                        #     fname = rootDir+'/'+dir_name1+'/'+ col_corr[col] +'/seg_'+str(i)+'_'+filename #dir_name2+'/seg_'+str(i)+'_'+filename
                        #     plt.imsave(fname, res)
                        #     i=i+1

def tests():
    img = './graphs_filtered/train/negative/reg_bar_graph3.png'
    graph = cv2.imread(img)
    graph = cv2.cvtColor(graph, cv2.COLOR_BGR2RGB)
    print(hsvRange(idColor(graph)))
#saveGraphs()
#tests()
#segmentImg()