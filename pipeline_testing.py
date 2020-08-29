import matplotlib.pyplot as plt
import numpy as np
from random import choice as randomchoice
import string
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
import math
from pipeline import *

IMG_HEIGHT = 480
IMG_WIDTH = 640

memoization = {}

def editfast(s,t):
    
    # change both to lower case to account for small case errors which are not important to distinguish between
    s = s.lower()
    t = t.lower()

    if (s,t) in memoization:
        return memoization[(s,t)]
    
    if s == "":
        return len(t)
    
    if t == "":
        return len(s)
    
    rtn = min([1 + editfast(s[1:], t), 1 + editfast(s, t[1:]), (s[ 0 ] != t[ 0 ]) + editfast(s[ 1 :], t[ 1 :])])
    
    memoization[(s,t)] = rtn
    
    return rtn

def get_random_string(length):
    # Random string with the combination of lower and upper case
    letters = string.ascii_letters
    res = ''.join(randomchoice(letters) for i in range(length))
    return res

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
        X1 = (a * Y1) + (b * Y2)
        X2 = (c * Y1) + (d * Y2)
    
    # calculate correlation
    corr = corr, _ = spearmanr(X1, X2) # spearman correlation
    return (X1,X2,corr)


# create multidata
def create_multiData(n, sN, train_val, seriesType, dcolor, dataStyle, verbose=0):
    
    #determine variables
    STcopy = seriesType
    possSeries = ['line', 'scatter'] #'bar']
    if sN == 1:
        possSeries.append('bar')
    possSeries = ['scatter']
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
    #posRGB = {(255,0,0):'r', (0,128,0):'g', (0,0,255):'b'}
    if dcolor == 'multi2':
        colors = ['y', 'c', 'm']#colors = ['y', 'c', 'm', 'r', 'g', 'b']
        #posRGB = {(0,255,255):'c', (255,0,255):'m', (255,255,0):'y'}#posRGB = {(0,255,255):'c', (255,0,255):'m', (255,255,0):'y', (255,0,0):'r', (0,128,0):'g', (0,0,255):'b'}
    
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
    fig, ax = plt.subplots()
    for i,var in enumerate(varArr):
        ((X1,X2,corrph),GT) = var
        if corrph >= 0.4:
            correlation = 'positive'
        elif corrph <= -0.4:
            correlation = 'negative'
        else:
            correlation = 'neutral'
        lbl = get_random_string(randint(3,12))
        label_to_corr_map[lbl] = correlation
        if GT == 'line':
            plt.plot(X1, X2, color=colArr[i], linestyle=LSarr[i], label=lbl)
        elif GT == 'scatter':
            ax.scatter(X1, X2, color=colArr[i], label=lbl)
        elif GT == 'bar':
            plt.bar(X1,X2, color=colArr[i], label=lbl)
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
    #plt.title(titlestr, loc=choice(tpos))
    tobj = ax.set_title(titlestr,loc=choice(tpos))
    fig.canvas.draw()
    
    

    # name the given graph
    fname = "images/" + "graph_" + str(n) + ".jpg"
    fig.savefig(fname)
    rend = fig.canvas.get_renderer()
    #print(rend)
    lbb = leg.get_window_extent(renderer=rend)
    #l_coords = leg.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    #xlbl_coords = ax.xaxis.get_window_extent(renderer=rend)#l_coords = leg.get_frame().get_bbox().bounds
    xlbl = ax.xaxis.get_label()
    ylbl = ax.yaxis.get_label()
    xlbb = xlbl.get_window_extent()
    ylbb = ylbl.get_window_extent()
    tbb = tobj.get_window_extent()

    plt.close()

    #fig.savefig(fname)
    return ("graph_" + str(n), X1s, X2s, titlestr, label_to_corr_map)#lbb, xlbb, ylbb, tbb)


def writeOutput(numGraphs):
    ground_truth = []
    ground_truth.append(("filename","x axis label","y axis label","title string","legend and correlation"))
    
    posSNs = [1,2,3]
    axis_hits = 0
    series_hits = 0
    hits = 0
    for i in range(0,numGraphs):
        leg_display_str = ""
        name,x1s,x2s,tstr,label_to_corr_map = create_multiData(i, choice(posSNs), "train", "random", "multi", "solid")
        iname = name + ".jpg"
        display_string = "images/" + iname + ", x axis: " + x1s + ", y axis: " + x2s + ", title: " + tstr
        leg_set = set()
        for key in label_to_corr_map:
            leg_set.add(key + ": " + label_to_corr_map[key])
            leg_display_str = leg_display_str + ", " + key + ": " + label_to_corr_map[key]
        ground_truth.append((iname, x1s, x2s, tstr, leg_display_str))
        #display_string = display_string + leg_display_str
        test_string,test_leg_set = process_img("images/" + iname)
        print(display_string)
        print(leg_set)
        print(test_string)
        print(test_leg_set)
        edds = editfast(display_string, test_string)
        if edds < 6:
            axis_hits = axis_hits + 1
        else:
            print('axis fail --------------------------------')
        
        oldsh = series_hits
        for elem1 in leg_set:
            for elem2 in test_leg_set:
                memoization = {}
                edsm = editfast(elem1,elem2)
                #print('edsm =' + str(edsm))
                if edsm < 3:
                    series_hits = series_hits + (1/len(leg_set))
                    break
        if oldsh == series_hits:
            print('series fail --------------------------------')
        if not bool(test_leg_set):
            print('empty set for legend')

    hits = (axis_hits + series_hits)/2
    print("axis score: " + str(axis_hits/numGraphs))
    print("series score: " + str(series_hits/numGraphs))
    score = (hits/numGraphs)*100
    print("total score: " + str(score) + "%")

    # write csv files
    with open('./ground_truth.csv', 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerows(ground_truth)
    f.close()

writeOutput(50)

