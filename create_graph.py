# import libraries
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import randint
from numpy.random import random
from numpy.random import randn
from numpy.random import choice
from numpy.random import seed
from numpy import cov
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import sys
import math
import os
import csv

def create_scatter_graph (n, train_val, multi, verbose=0): 

    # seed random number generator -- uncomment line below to turn seeding on
    # seed(n)

    # generate correlation -- equations adapted from: http://hosting.astro.cornell.edu/~cordes/A6523/GeneratingCorrelatedRandomVariables.pdf
    sign = [-1,1]
    correlation = choice(sign) * random()
    if verbose:
        print(correlation)
    Y1 = randn(1000)
    Y2 = randn(1000)
    phi = (0.5) * math.asin(correlation)
    a = math.cos(phi)
    b = math.sin(phi)
    c = math.sin(phi)
    d = math.cos(phi)
    X1 = (a * Y1) + (b * Y2)
    X2 = (c * Y1) + (d * Y2)

    # Calculate Correlation -- will be very similar to intended correlation for large data inputs
    # Following code adapted from: https://machinelearningmastery.com/how-to-use-correlation-to-understand-the-relationship-between-variables/
    # ------------------------------------------
    # calculate covariance matrix
    covariance = cov(X1, X2)
    if verbose:
        print(covariance)

    # calculate Pearson's correlation
    corr, _ = pearsonr(X1, X2)
    if verbose:
        print('Pearsons correlation: %.3f' % corr)

    # calculate spearman's correlation
    corr, _ = spearmanr(X1, X2)
    if verbose:
        print('Spearmans correlation: %.3f' % corr)
    # ------------------------------------------

    # plot
    colors = ['r', 'g', 'b'] # add other colors to see if it affects learning
    if multi:
        col = choice(colors)
    else:
        col = 'b'
    fig, ax = plt.subplots()
    ax.scatter(X1, X2, color=col)



    if corr >= 0.4:
        s = "positive"
        simpcorr = 1
    elif corr <= -0.4:
        s = "negative"
        simpcorr = -1
    else:
        s = "neutral"
        simpcorr = 0

    # name the given graph
    fname = "graphs_filtered/" + train_val + "/" + s + "/" + "scatter_graph" + str(n) + ".png"

    # create ordered pair
    ret = ("graph" + str(n), corr, simpcorr) #(fname, correlation, rounded correlation)

    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title(s + " Correlation")
    #plt.legend()
    #print(fname)
    #plt.show()
    fig.savefig(fname)
    return ret


# line graph generator
def create_line_graph (n, train_val, multi, lineType, verbose=0):

    # determine variables
    # slope and intercept
    sign = [-1,1]
    m = choice(sign) * random() # determine slope
    b = choice(sign) * randint(0,5) * random() # determine intercept
    
    # colors
    colors = ['r', 'g', 'b'] # add other colors to see if it affects learning
    if multi:
        col = choice(colors)
    else:
        col = 'b'

    # lineStyles
    lineStyles = ['solid', 'dotted', 'dashed', 'dashdot']
    if lineType:
        lineStyle = choice(lineStyles)
    else:
        lineStyle = 'solid'
    
    # determine correlation
    if m >= 0.4:
        correlation = 'positive'
    elif m <= -0.4:
        correlation = 'negative'
    else:
        correlation = 'neutral'


    # name the given graph
    fname = "graphs_filtered/" + train_val + "/" + correlation + "/" + "line_graph" + str(n) + ".png"

    # plot
    fig, ax = plt.subplots()
    x = np.linspace(start = 0, stop = 10)
    ax.plot(x, (m*x)+b, linestyle=lineStyle, color=col)
    ax.set_xlim([0, 10])
    ax.set_ylim([-10, 10])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(correlation + " Correlation")
    plt.show
    fig.savefig(fname)
    return ("line_graph" + str(n), float(m), correlation)


# create the training data
def create_data(size, train, val, graphType, lineType, v, directory):
    cwd=os.getcwd()
    if(cwd!=directory):
        print("error: create_data called from wrong directory")
    else:
        # create the appropriate training, validation, and correlation directories
        # ----------------------------------------

        # create graphs_filtered
        path = "./graphs_filtered"
        try:
            os.mkdir(path)
        except OSError:
            print ("Warning: Creation of the directory %s failed, might already exist" % path)

        # create training and validation directories
        correlations = ["positive", "negative", "neutral"]
        for correlation in correlations:
            train_path = "./graphs_filtered/train/" + correlation
            try:
                os.makedirs(train_path)
            except OSError:
                print ("Warning: Creation of the directory %s failed, might exist already" % train_path)
            train_path = "./graphs_filtered/validation/" + correlation
            try:
                os.makedirs(train_path)
            except OSError:
                print ("Warning: Creation of the directory %s failed, might exist already" % train_path)

        # ----------------------------------------



        # initialize the list (array)
        graphs = list()

        # create a file with the graphnames and correlations
        graphs.append(("Title", "Correlation", "Rounded Correlation"))
        if graphType == "scatter":
            for i in range (0, size):
                graphs.append(create_scatter_graph(i+1, "train", train, v))
                graphs.append(create_scatter_graph(i+1, "validation", val, v))
        elif graphType == "line":
            for i in range (0, size):
                graphs.append(create_line_graph(i+1, "train", train, lineType, v))
                graphs.append(create_line_graph(i+1, "validation", val, lineType, v))
        else:
            print("error: create_data called with wrong graphType")

        with open('graph_info.csv', 'w') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerows(graphs)
        f.close()

        # print success to console
        print("create_training_data: successful")

        # return the current path (this will be used for the image classication program)
        return os.getcwd()