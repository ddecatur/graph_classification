from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import colorsys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from predict import predictCategory


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

# determine appropriate k
def find_k(x):
  pred = predictCategory(x,'series_class_model_v2.h5',[1,2,3])
  return pred+2

# Implement the K Means Clustering
def KMeansCluster (imageArr,img):
    k = find_k(img) # 3 # hard coding this to save time for now
    clt = KMeans(n_clusters = k)
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
#tests()