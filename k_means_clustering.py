from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import warnings
from predict import predictCategory
from kneed import KneeLocator


# clustering code adapted from https://www.pyimagesearch.com/2014/05/26/opencv-python-k-means-color-clustering/
def clusterCounts(clt):
	numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
	(hist, _) = np.histogram(clt.labels_, bins = numLabels)
	hist = hist.astype("float")
	hist /= hist.sum()
	return hist


def calculate_WSS(points, kmax): # aka the elbow method -- code adapted from #https://medium.com/analytics-vidhya/how-to-determine-the-optimal-k-for-k-means-708505d204eb
  sse = []
  for k in range(1, kmax+1):
    kmeans = KMeans(n_clusters = k).fit(points)
    centroids = kmeans.cluster_centers_
    pred_clusters = kmeans.predict(points)
    curr_sse = 0
    
    # calculate square of Euclidean distance of each point from its cluster center and add to current WSS
    for i in range(len(points)):
      curr_center = centroids[pred_clusters[i]]
      curr_sse += (points[i, 0] - curr_center[0]) ** 2 + (points[i, 1] - curr_center[1]) ** 2
      
    sse.append(curr_sse)
  return sse


def sil_score(points, kmax):
  sil = []

  # dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
  for k in range(2, kmax+1):
    kmeans = KMeans(n_clusters = k).fit(points)
    labels = kmeans.labels_
    sil.append(silhouette_score(points, labels, metric = 'euclidean'))
  
  idx = 2
  maxsil = 0
  for i in range(len(sil)):
    if sil[i] > maxsil:
      maxsil = sil[i]
      idx = i+2
  return idx


def nc_in_set(h,s,dic):
  minD = 10 # max color distance
  for key in dic:
    lh = list()
    uh = list()
    low = h-minD
    high = h+minD
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
    for lb,ub in zip(lh,uh):
      if (lb <= key[0] <= ub) and abs(int(s)-int(key[1])) < 10:
        return key
  
  return (h,s)


def pair_avg(p1,p2):
  x1,y1 = p1
  x2,y2 = p2
  return ((int(x1)+int(x2))//2, (int(y1)+int(y2))//2)


def num_diff_cols(img):
  img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  numCols = 0
  mem = {}
  for i in range(img.shape[0]//5):
    for j in range(img.shape[1]//5):
        h,s,v = img[i*5][j*5]
        key = nc_in_set(h, s, mem)
        if key not in mem:
          mem[key] = 1
        else:
          temp = mem[key]
          del mem[key]
          mem[pair_avg(key,(h,s))] = temp+1
  for elem in mem:
    if mem[elem] > 30:
      numCols += 1
  
  return numCols


def elbowM(arr, kneedleBasic=False):
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    arr = arr.reshape((arr.shape[0] * arr.shape[1], 3))
    maxk = 8
    y = []
    for i in range(1,maxk+1):
      kmeans = KMeans(n_clusters=i)
      kmeans.fit(arr)
      y.append(kmeans.inertia_)
    x = range(1, len(y)+1)
    kn = KneeLocator(x, y, S=3.0, online=True, curve='convex', direction='decreasing')
    
    # for basic kneedle usage
    if kneedleBasic:
      return kn.knee
    
    if kn.y_normalized[kn.knee-1] < 0.985:#used to be 99
      return kn.knee+1
    else:
      return kn.knee
  


# determine appropriate k (depreciated)
def find_k(x):
  pred = predictCategory(x,'models/series/series_class_model_v3.h5',[1,2,3]) #'series_class_model_mDS_DC_93acc.h5' 'series_class_model_v2.h5'
  return pred+1

# Implement the K Means Clustering
def KMeansCluster (imageArr,img):#, setK):
    im = Image.fromarray(imageArr)
    im.save('k_placeholder.png')
    k = elbowM(imageArr)#num_diff_cols(imageArr)#sil_score(imageArr.reshape((imageArr.shape[0] * imageArr.shape[1], 3)), 3)#find_k(img)
    print("k is: " + str(k))
    clt = KMeans(n_clusters = k)
    imageArr = imageArr.reshape((imageArr.shape[0] * imageArr.shape[1], 3))
    clt.fit(imageArr)
    return clt

# these next two are testing functions taken from https://www.pyimagesearch.com/2014/05/26/opencv-python-k-means-color-clustering/
def centroid_histogram(clt):
	numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
	(hist, _) = np.histogram(clt.labels_, bins = numLabels)
	hist = hist.astype("float")
	hist /= hist.sum()
	return hist

def plot_colors(hist, centroids):
	bar = np.zeros((50, 300, 3), dtype = "uint8")
	startX = 0
	for (percent, color) in zip(hist, centroids):
		endX = startX + (percent * 300)
		cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
			color.astype("uint8").tolist(), -1)
		startX = endX
	return bar

def tests():
    img = './graphs_filtered/testttttt.png'
    graph = cv2.imread(img)
    graph = cv2.cvtColor(graph, cv2.COLOR_BGR2RGB)
    graph = graph.reshape((graph.shape[0] * graph.shape[1], 3))
    clt = KMeansCluster(graph)
    hist = centroid_histogram(clt)
    bar = plot_colors(hist, clt.cluster_centers_)
    # show our color bart
    plt.figure()
    plt.axis("off")
    plt.imshow(bar)
    plt.show()