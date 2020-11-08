# import libraries
import matplotlib.pyplot as plt
import numpy as np
from random import choice as randomchoice
from numpy.random import randint
from numpy.random import random
from numpy.random import randn
from numpy.random import choice
from numpy.random import seed
from numpy import cov
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from seg_img import *
import sys
import math
import os
import csv
import math
import string

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
        
        # # adjust intercept to make bar graph not cross y axis
        # minVal = min(X2)
        # #print(minVal)
        # b = 0 - minVal
        # X2 = (m * X1) + b + delta
    elif dataType == 'scatter':
        sign = [-1,1]
        correlation = choice(sign) * random()
        Y1 = randn(1000)
        Y2 = randn(1000)
        phi = (0.5) * math.asin(correlation)
        a = math.cos(phi)
        b = math.sin(phi)
        c = math.sin(phi)
        d = math.cos(phi)
        intercept = choice(sign) * randint(0,10) * random()
        X1 = (a * Y1) + (b * Y2)
        X2 = (c * Y1) + (d * Y2) + intercept
    
    # calculate correlation
    corr = corr, _ = spearmanr(X1, X2) # spearman correlation
    return (X1,X2,corr)

def col_dist(c1,c2):
    (r,g,b) = c1
    (R,G,B) = c2
    return math.sqrt((r-R)**2 + (g-G)**2 + (b-B)**2)

def find_nearest_col(color, dic):
    dist = list()
    (R,G,B) = color
    minD = 442 # max color distance
    minC = 'empty'
    for elem in dic:
        d = col_dist(elem,color) #math.sqrt((r-R)^2 + (g-G)^2 + (b-B)^2)
        if d < minD:
            minD = d
            minC = dic.get(elem)
    return minC
    
    
def get_random_string(length):
    # Random string with the combination of lower and upper case
    letters = string.ascii_letters
    res = ''.join(randomchoice(letters) for i in range(length))
    return res

# # create multi data graph
def create_multiData(n, sN, train_val, seriesType, dcolor, dataStyle, model, verbose=0):
    
    #determine variables
    STcopy = seriesType
    possSeries = ['line', 'scatter', 'bar']
    #sN = 3
    # if sN == 1:
    #     possSeries.append('bar')
    #possSeries = ['bar']
    varArr = np.empty (sN, tuple)
    for i in range (0,sN):
        if STcopy == 'random':
            STcopy = choice(possSeries)
        if STcopy == 'line':
            varArr[i] = (genData('line'),'line')
        elif STcopy == 'bar':
            varArr[i] = (genData('bar'),'bar')
        elif STcopy == 'scatter':
            varArr[i] = (genData('scatter'),'scatter')

    # colors
    colors = ['r', 'g', 'b'] # add other colors to see if it affects learning
    posRGB = {(255,0,0):'r', (0,128,0):'g', (0,0,255):'b'}
    if dcolor == 'multi2':
        colors = ['y', 'c', 'm']#colors = ['y', 'c', 'm', 'r', 'g', 'b']
        posRGB = {(0,255,255):'c', (255,0,255):'m', (255,255,0):'y'}#posRGB = {(0,255,255):'c', (255,0,255):'m', (255,255,0):'y', (255,0,0):'r', (0,128,0):'g', (0,0,255):'b'}
    #colors = ['y', 'c', 'm', 'r', 'g', 'b']

    #posRGB = {(0,255,255):'c', (255,0,255):'m', (255,255,0):'y', (255,0,0):'r', (0,128,0):'g', (0,0,255):'b'}

    copyC = colors
    colArr = np.empty(sN, str)
    for i in range (0, sN):
        elem = choice(copyC)
        colArr[i] = elem
        copyC.remove(elem)


    # lineStyles
    lineStyles = ['solid', 'dotted', 'dashed', 'dashdot']
    LSarr = list()
    if dataStyle == 'multi':
        for i in range (0, sN):
            LSarr.append(choice(lineStyles))
    else:
        for i in range (0, sN):
            LSarr.append('solid')


    # plot
    label_to_corr_map = {}
    correlation = {}
    corr = list()
    fig, ax = plt.subplots()
    plot_options = plt.style.available
    if 'grayscale' in plot_options:
        plot_options.remove('grayscale')
    if 'dark_background' in plot_options:
        plot_options.remove('dark_background')
    # if 'dark_background' in plot_options:
    #     plot_options.remove('dark_background')
    # if train_val == 'train':
    #     plt.style.use('classic')
    # else:
    #     plt.style.use('ggplot')
    plt.style.use('default')
    style = choice(plot_options)
    plt.style.use(style)
    for i,var in enumerate(varArr):
        ((X1,X2,corrph),GT) = var
        if corrph >= 0.4:
            correlation[colArr[i]] = 'positive'
        elif corrph <= -0.4:
            correlation[colArr[i]] = 'negative'
        else:
            correlation[colArr[i]] = 'neutral'
        lbl = get_random_string(randint(3,12))
        label_to_corr_map[lbl] = correlation[colArr[i]]
        if GT == 'line':
            if model == 'g':
                plt.plot(X1, X2, linestyle=LSarr[i], label=lbl, color=colArr[i])     
            else:
                plt.plot(X1, X2, linestyle=LSarr[i], label=lbl) # color=colArr[i], 
        elif GT == 'scatter':
            if model == 'g':
                ax.scatter(X1, X2, label=lbl, color=colArr[i])     
            else:
                ax.scatter(X1, X2, label=lbl) # color=colArr[i], 
        elif GT == 'bar':
            if model == 'g':
                w = 0.8 #* len(varArr)
                ax.bar((len(varArr)*X1)+(w*i), X2, width=w, align='center', label=lbl, color=colArr[i])
            else:
                w = 0.8 #* len(varArr)
                ax.bar((len(varArr)*X1)+(w*i), X2, width=w, align='center', label=lbl) # color=colArr[i], 
        else:
            raise ValueError('graph type not recognized')

    # randomize label and title positions and  strings
    ylabelpos = ['left', 'right']
    xlabelpos = ['top', 'bottom']
    tpos = ylabelpos + ['center']
    X1s = get_random_string(randint(3,12))
    X2s = get_random_string(randint(3,12))
    titlestr = get_random_string(randint(3,12))
    plt.xlabel(X1s, labelpad=randint(2,10))
    plt.ylabel(X2s, labelpad=randint(2,10))
    ax.xaxis.set_label_position(choice(xlabelpos))
    ax.yaxis.set_label_position(choice(ylabelpos))
    leg = ax.legend()
    tobj = ax.set_title(titlestr,loc=choice(tpos))
    fig.canvas.draw()

    # name the given graph
    if model == 'g':
        fname = "placeholder.png"
        fig.savefig(fname)
        segImg = segmentImg(fname)
        for i,(img,col) in enumerate(segImg):
            closeCol = find_nearest_col(col,posRGB)
            if closeCol in correlation:
                corrstr = correlation[closeCol]
                fname = "graphs_filtered/" + train_val + "/" + corrstr + "/" + "seg_" + corrstr + "_" + closeCol + str(i) + "_" + seriesType + "_graph" + str(n) + ".png" # changed to jpg
                plt.imsave(fname,img)
            else:
                print('closest color not found')
                print(col)
        plt.close('all')
    elif model == 's':
        fname = "series_filtered/" + train_val + "/" + str(sN) + "/" + str(sN) + "_graph" + str(n) + ".png" # changed to jpg
        fig.savefig(fname)
        img = cv2.imread(fname)
        img = sat_thresh_filter(img,30)
        im = Image.fromarray(img)
        im.save(fname)
        plt.close()
    else:
        # name the given graph
        fname = "images/" + "graph_" + str(n) + ".png"
        fig.savefig(fname)
        plt.close()
        return ("graph_" + str(n), X1s, X2s, titlestr, label_to_corr_map)
    
    return ("line_graph" + str(n), float(corrph), "placeholder")


# create the training data
def create_training_data(size, seriesNum, graphType, color, dataStyle, v, directory):
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
        
        
        if seriesNum=='multi':
            sNop = ["1", "2", "3"]
            for i in range (0,size):
                sNopC = int(choice(sNop))
                create_multiData(i+1, sNopC, "train", train_gt, train_col, train_ds, 'g', v)
            for i in range (0,size):
                sNopC = int(choice(sNop))
                create_multiData(i+1, sNopC, "validation", val_gt, val_col, val_ds, 'g', v)

        else:
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
            create_multiData(i, sNopC, 'train', 'random', 'multi', dataStyle, 's')
            create_multiData(i, sNopC, 'validation', 'random', 'multi', dataStyle, 's')