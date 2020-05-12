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
    if multi == 'multi':
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
    fname = "graphs_filtered/" + train_val + "/" + s + "/" + "reg_scatter_graph" + str(n) + ".png"

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

# create perturbed data
def genData(dataType):
    # determine variables
    if dataType == 'line':
        # slope and intercept
        sign = [-1,1]
        m = choice(sign) * random() # determine slope
        b = choice(sign) * randint(0,25) * random() # determine intercept
        delta = np.random.uniform(-50,50, size=(100,))
        X1 = np.arange(100)
        X2 = (m * X1) + b + delta
    elif dataType == 'bar':
        # slope and intercept
        sign = [-1,1]
        m = choice(sign) * random() # determine slope
        #b = choice(sign) * randint(0,50) * random() # determine intercept
        delta = np.random.uniform(-15,15, size=(50,))
        X1 = np.arange(50)
        X2 = (m * X1) + delta
        
        # adjust intercept to make bar graph not cross y axis
        minVal = min(X2)
        #print(minVal)
        b = 0 - minVal
        X2 = (m * X1) + b + delta
    
    # calculate correlation
    corr = corr, _ = spearmanr(X1, X2) # spearman correlation
    return (X1,X2,corr)

# line graph generator
def create_line_graph (n, train_val, multi, lineType, verbose=0):

    #determine variables
    (X1, X2, corr) = genData('line')
    
    # colors
    colors = ['r', 'g', 'b'] # add other colors to see if it affects learning
    if multi == 'multi':
        col = choice(colors)
    else:
        col = 'b'

    # lineStyles
    lineStyles = ['solid', 'dotted', 'dashed', 'dashdot']
    if lineType == 'multi':
        lineStyle = choice(lineStyles)
    else:
        lineStyle = 'solid'
    
    # determine correlation
    if corr >= 0.4:
        correlation = 'positive'
    elif corr <= -0.4:
        correlation = 'negative'
    else:
        correlation = 'neutral'


    # name the given graph
    fname = "graphs_filtered/" + train_val + "/" + correlation + "/" + "reg_line_graph" + str(n) + ".png"

    # plot
    fig, ax = plt.subplots()
    plt.plot(X1,X2, color = col, linestyle=lineStyle)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title(correlation + " Correlation")
    #plt.show
    fig.savefig(fname)
    return ("line_graph" + str(n), float(corr), correlation)

# Bar graph generator
def create_bar_graph (n, train_val, multi, barType, verbose=0):

    # determine variables
    (X1,X2,corr) = genData('bar')
    
    # colors
    colors = ['r', 'g', 'b'] # add other colors to see if it affects learning
    if multi == 'multi':
        col = choice(colors)
    else:
        col = 'b'

    # lineStyles
    barStyles = ['solid', 'dotted', 'dashed', 'dashdot']
    if barType == 'multi':
        barStyle = choice(barStyles)
    else:
        barStyle = 'solid'
    
    # determine correlation
    if corr >= 0.5:
        correlation = 'positive'
    elif corr <= -0.5:
        correlation = 'negative'
    else:
        correlation = 'neutral'


    # name the given graph
    fname = "graphs_filtered/" + train_val + "/" + correlation + "/" + "reg_bar_graph" + str(n) + ".png"

    # plot
    fig, ax = plt.subplots()
    plt.bar(X1,X2, color = col, linestyle=barStyle)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title(correlation + " Correlation")
    #plt.show
    fig.savefig(fname)
    return ("bar_graph" + str(n), float(corr), correlation)


# create multi data graph
def create_multiData(n, sN, train_val, lineType, model, verbose=0):
    #determine variables
    
    varArr = np.empty (sN, tuple)
    for i in range (0,sN):
        varArr[i] = genData('line')
        
    (X1, X2, corr1) = varArr[0]
    # (Y1, Y2, corr2) = genData('line')

    # colors
    colors = np.array(['r', 'g', 'b']) # add other colors to see if it affects learning
    colArr = np.empty(sN, str)
    for i in range (0, sN):
        colArr[i] = colors[i]
    # col1 = 'b'#choice(colors) -- for now make it blue for simplicity
    # col2 = 'r'#choice(colors)
    #while col2 == col2: # prevent the two colors from being the same
    #    col2 = choice(colors)


    # lineStyles
    lineStyles = ['solid', 'dotted', 'dashed', 'dashdot']
    if lineType == 'multi':
        lineStyle = choice(lineStyles)
    else:
        lineStyle = 'solid'
    
    # determine correlations
    if corr1 >= 0.4:
        correlation1 = 'positive'
    elif corr1 <= -0.4:
        correlation1 = 'negative'
    else:
        correlation1 = 'neutral'

    # name the given graph
    if model == 'g':
        fname = "graphs_filtered/" + train_val + "/" + correlation1 + "/" + "reg_line_graph" + str(n) + ".png"
    else:
        fname = "series_filtered/" + train_val + "/" + str(sN) + "/" + str(sN) + "_reg_line_graph" + str(n) + ".png"

    # plot
    fig, ax = plt.subplots()
    for i,var in enumerate(varArr):
        (X1,X2,corr) = var
        plt.plot(X1, X2, color=colArr[i], linestyle=lineStyle)
    
    # plt.plot(X1,X2, color = col1, linestyle=lineStyle)
    # plt.plot(Y1,Y2, color = col2, linestyle=lineStyle)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title(str(sN) + " Series")
    #plt.show
    fig.savefig(fname)
    return ("line_graph" + str(n), float(corr1), correlation1)

# create the training data
def create_training_data(size, graphType, color, dataStyle, v, directory):
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

        # match tuples
        (train_gt, val_gt) = graphType
        (train_col, val_col) = color
        (train_ds, val_ds) = dataStyle
        #test = [train_gt, val_gt, train_col, val_col, train_ds, val_ds]
        dt = ['scatter','line','bar'] # list of possible data types
        errorMsg = False

        # Training Data
        orig_gt = train_gt
        graphs.append(("Title", "Correlation", "Rounded Correlation"))
        for i in range (0,size):
            if orig_gt == "random":
                train_gt = choice(dt)
            if train_gt == "scatter":
                graphs.append(create_scatter_graph(i+1, "train", train_col, v))
            elif train_gt == "line":
                graphs.append(create_line_graph(i+1, "train", train_col, train_ds, v))
            elif train_gt == "bar":
                graphs.append(create_bar_graph(i+1, "train", train_col, train_ds, v))
            else:
                errorMsg = True
        
        # Validation Data
        orig_gt = val_gt
        for i in range (0, size):
            # choose graph type if random
            if orig_gt == "random":
                val_gt = choice(dt)
            if val_gt == "scatter":
                graphs.append(create_scatter_graph(i+1, "validation", val_col, v))
            elif val_gt == "line":
                graphs.append(create_line_graph(i+1, "validation", val_col, val_ds, v))
            elif val_gt == "bar":
                graphs.append(create_bar_graph(i+1, "validation", val_col, val_ds, v))
            else:
                errorMsg = True
        if errorMsg == True:
            print("error: create_data called with wrong graphType (validation)")    
    
        # create CSV file with graph names and correlations
        with open('graph_info.csv', 'w') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerows(graphs)
        f.close()

        # print success to console
        print("create_training_data: successful")

        # return the current path (this will be used for the image classication program)
        return os.getcwd()

def train_series_class(size, sN, dataStyle, directory):
    cwd=os.getcwd()
    if(cwd!=directory):
        print("error: create_data called from wrong directory")
    else:
        # ----------------------------------------
        path = "./series_filtered"
        try:
            os.mkdir(path)
        except OSError:
            print ("Warning: Creation of the directory %s failed, might already exist" % path)

        # create training and validation directories
        sNop = ["1", "2", "3"]
        for n in sNop:
            train_path = "./series_filtered/train/" + n
            try:
                os.makedirs(train_path)
            except OSError:
                print ("Warning: Creation of the directory %s failed, might exist already" % train_path)
            train_path = "./series_filtered/validation/" + n
            try:
                os.makedirs(train_path)
            except OSError:
                print ("Warning: Creation of the directory %s failed, might exist already" % train_path)

        # ----------------------------------------

        for i in range (0, size):
            sNopC = int(choice(sNop))
            create_multiData(i, sNopC, 'train', dataStyle, 's')
            create_multiData(i, sNopC, 'validation', dataStyle, 's')


#test_multiData()