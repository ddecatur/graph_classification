from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import colorsys
import cv2
import numpy as np
import matplotlib.pyplot as plt


# clustering code adapted from https://www.pyimagesearch.com/2014/05/26/opencv-python-k-means-color-clustering/
def clusterCounts(clt):
	# grab the number of different clusters and create a histogram
	# based on the number of pixels assigned to each cluster
	numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
	(hist, _) = np.histogram(clt.labels_, bins = numLabels)
	# normalize the histogram, such that it sums to one
	hist = hist.astype("float")
	hist /= hist.sum()
	# return the histogram
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
    print("test2")
    kmax = 4
    #for k in range(2, kmax+1):
    #    kmeans = KMeans(n_clusters = k).fit(x)
    #    labels = kmeans.labels_
    #    sil.append(silhouette_score(x, labels, metric = 'euclidean')) 
    sse = calculate_WSS(x, kmax)
    sseMax = 0
    print("test3")
    for i in range(2,kmax+1):
        if sse[i-2] > sseMax:
            silMax = sse[i-2]
            idealK = i
    return idealK

# Implement the K Means Clustering
def KMeansCluster (imageArr):
    k = 3#find_k(imageArr) -- hard coding this to save time for now
    clt = KMeans(n_clusters = k)
    clt.fit(imageArr)
    return clt

# these next two are testing functions taken from https://www.pyimagesearch.com/2014/05/26/opencv-python-k-means-color-clustering/
def centroid_histogram(clt):
	# grab the number of different clusters and create a histogram
	# based on the number of pixels assigned to each cluster
	numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
	(hist, _) = np.histogram(clt.labels_, bins = numLabels)
	# normalize the histogram, such that it sums to one
	hist = hist.astype("float")
	hist /= hist.sum()
	# return the histogram
	return hist

def plot_colors(hist, centroids):
	# initialize the bar chart representing the relative frequency
	# of each of the colors
	bar = np.zeros((50, 300, 3), dtype = "uint8")
	startX = 0
	# loop over the percentage of each cluster and the color of
	# each cluster
	for (percent, color) in zip(hist, centroids):
		# plot the relative percentage of each cluster
		endX = startX + (percent * 300)
		cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
			color.astype("uint8").tolist(), -1)
		startX = endX
	
	# return the bar chart
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