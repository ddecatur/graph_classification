import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from k_means_clustering import *
import math

setK = None

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

def blur(img):
    return cv2.GaussianBlur(img,(5,5),0)

def median_blur(img):
    return cv2.medianBlur(img,5)

def descritize(img):
    data = np.array(img)
    data = data / 255.0
    (h,w,c) = data.shape
    shape = data.shape
    print(data.shape)
    data = data.reshape(h*w, c)
    print(data.shape)
    kmeans = KMeans(n_clusters=13)
    kmeans.fit(data)
    new_colors = kmeans.cluster_centers_[kmeans.predict(data)]
    print(shape)
    img_recolored = new_colors.reshape(shape)
    img_recolored = 255 * img_recolored # Now scale by 255
    img_recolored = img_recolored.astype(np.uint8)
    display_img = Image.fromarray(img_recolored)
    display_img.show()

    return img_recolored

def sharpen(img):
    kernal_h = np.array([[-1,-1,-1], 
                       [-1, 9,-1],
                       [-1,-1,-1]])
    return cv2.filter2D(img, -1, kernal_h)

def eximg(img):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            (r,g,b) = img[i][j]
            img[i][j] = (0,0,b)
    return img

def sat_max(*cols):
    max_s = 256
    max_col = None
    for (h,s,v) in cols:
        if s < max_s:
            max_s = s
            max_col = (h,s,v)
    return max_col

def sat_thresh_filter(img,thresh):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    new_img = np.empty(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            (h,s,v) = img[i][j]
            if s > thresh:
                new_img[i][j] = (h,255,255)
            else:
                new_img[i][j] = (h,0,255)
    new_img = new_img.astype(np.uint8)
    new_img = cv2.cvtColor(new_img, cv2.COLOR_HSV2RGB)
    return new_img

def orange_to_red(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    new_img = np.empty(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            (h,s,v) = img[i][j]
            if 5<=h and h<=30:
                new_img[i][j] = (0,s,v)
            else:
                new_img[i][j] = (h,s,v)
    new_img = new_img.astype(np.uint8)
    new_img = cv2.cvtColor(new_img, cv2.COLOR_HSV2RGB)
    return new_img

def max_sat_filter(img,n):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    new_img = np.empty(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            max_col = tuple(img[i][j])
            h,s,v = max_col
            if s < 240:
            #if tuple(img[i][j]) != (0,0,255): # to leave white tiles white
                for ii in range(-n,n):
                    for jj in range(-n,n):
                        if i+ii in range(img.shape[0]) and j+jj in range(img.shape[1]):
                            max_col = sat_max(max_col, tuple(img[i+ii,j+jj]))
            new_img[i][j] = max_col
    new_img = new_img.astype(np.uint8)
    new_img = cv2.cvtColor(new_img, cv2.COLOR_HSV2RGB)
    return new_img



# def Dale_filter(img, n):
#     '''
#         For n pixels, fit line line
#         Make all pixels in left half leftmost pixel of line * r^2 + current pixel * 1-r^2
#         Do horizontally then vertically
#     '''
#     for i in range(img.shape[])

def loc_descritize(img, n):
    new = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            ii = min(((i//n)*n) + n//2, img.shape[0]-1)
            jj = min(((j//n)*n) + n//2, img.shape[1]-1)
            img[i][j] = img[ii][jj]
    return img

def HOF(img):
    '''
    horz 2d filter -> h_notav
    vert 
    for each pixel, if val == 0,
    '''
    kernal_h = np.array([[0.5,-1,0.5], 
                       [0.5,-1,0.5],
                       [0.5,-1,0.5]])
    kernal_v = np.array([[0.5,0.5,0.5], 
                       [-1,-1,-1],
                       [0.5,0.5,0.5]])
    horz = cv2.filter2D(img, -1, kernal_h) / 255.0
    vert = cv2.filter2D(img, -1, kernal_v) / 255.0
    for i in range(img.shape[0]-1):
        for j in range(img.shape[1]-1):
            img[i][j] = (img[i][j] * abs(horz[i][j])) + (img[i][j+1] * abs(1-horz[i][j]))

    for i in range(img.shape[0]-1):
        for j in range(img.shape[1]-1):
            img[i][j] = (img[i][j] * abs(vert[i][j])) + (img[i+1][j] * abs(1-vert[i][j]))
    
    return img


def isGrayish(color):
    (r,g,b) = color
    ac = (r+g+b)/3
    cd = col_dist((ac,ac,ac), color)
    if cd < 20:
        return True
    else:
        return False

# type: img path --> list of color ranges
def idColor(image):
    rangeList = list()
    
    ogImg = image
    # convert image
    image = cv2.imread(image)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # convert color
    #image = median_blur(image)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #image = descritize(image)
    image = sat_thresh_filter(image,40)
    #image = orange_to_red(image)

    clt = KMeansCluster(image,ogImg)#,setK)
    hist = clusterCounts(clt)
    posRGB = {(255,0,0):'r', (0,128,0):'g', (0,0,255):'b', (0,0,0):'black', (255,255,255):'white'}
    # determine which colors to segment out
    colList = list()
    maxCol = max(hist) # identify the background color as the most common color so it can be removed
    for (percent, color) in zip(hist, clt.cluster_centers_):
        if (percent != maxCol and isGrayish(color)==False):
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
    sl = max(s-25,0)
    vl = max(v-40,76.5) # should be -40 # 
    su = min(s+25,255) # should be +10
    vu = min(v+40,255)
    for i,bound in enumerate(lh):
        rtn.add(((bound, sl, vl), (uh[i], su, vu)))
    return rtn
    
    

# given a list of rgb colors return a list of hsv ranges
def hsvRange (rgbList):
    rangeList = list()
    for (r,g,b) in rgbList:
        (h,s,v) = rgb_hsv(r,g,b)
        rangeList.append(find_ranges(h,s,v))

    return rangeList


def segmentImg(img, fixed_k=None):
    '''
    Givn an image path, segment out colors from that image
    '''
    global setK
    setK = fixed_k

    # read in image
    graph = cv2.imread(img)
    #graph = median_blur(graph)
    #graph = cv2.cvtColor(graph, cv2.COLOR_BGR2HSV)
    #graph = descritize(graph)
    graph = sat_thresh_filter(graph,40)
    #graph = orange_to_red(graph)
    #graph = cv2.cvtColor(graph, cv2.COLOR_BGR2RGB) # convert color from BGR to RGB
    #graph = cv2.cvtColor(graph, cv2.COLOR_HSV2RGB)
    
    colList = idColor(img)
    colRangeList = hsvRange(colList)

    # convert to hsv image type
    hsv_graph = cv2.cvtColor(graph, cv2.COLOR_RGB2HSV)
    results = list()
    for i,rngSet in enumerate(colRangeList):
        masks = list()
        for (ll,ul) in rngSet:
            masks.append(cv2.inRange(hsv_graph, ll, ul))
        maskFinal = masks[0]
        for j in range(1,len(masks)):
            maskFinal = maskFinal + masks[j]
        results.append((cv2.bitwise_and(graph, graph, mask=maskFinal), colList[i]))

    return results