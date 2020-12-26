from pipeline import dist
from graph_classification import graph_classification
import numpy as np
import cv2
import csv
import os
import warnings
import glob
import yaml
from PIL import Image
from predict import *
from create_graph import create_multiData, find_nearest_col
from numpy.random import choice
from seg_img import * #sat_thresh_filter, segmentImg
from k_means_clustering import elbowM, find_k
from clean import clean
from ocr import *


# Helper functions

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

def create_filtered_dirs(dirname):
    # create the appropriate training, validation, and correlation directories
        # ----------------------------------------

        # create graphs_filtered
        path = "./" + dirname #"./graphs_filtered"
        try:
            os.mkdir(path)
        except OSError:
            print ("Warning: Creation of the directory %s failed, might already exist" % path)

        # create training and validation directories
        correlations = ["positive", "negative", "neutral"]
        for correlation in correlations:
            train_path = "./" + dirname + "/train/" + correlation
            try:
                os.makedirs(train_path)
            except OSError:
                print ("Warning: Creation of the directory %s failed, might exist already" % train_path)
            train_path = "./" + dirname + "/validation/" + correlation
            try:
                os.makedirs(train_path)
            except OSError:
                print ("Warning: Creation of the directory %s failed, might exist already" % train_path)

        # ----------------------------------------

def resave_wild_graphs_for_classification(train_val, outputDir, traintest=None):
    posRGB = {(255,0,255):'purple', (255,0,0):'red', (0,128,0):'green', (0,0,255):'blue', (255,140,0):'orange'} #(0,255,255):'cyan', (255,255,0):'yellow', 
    with open('outside_data_labels.yaml') as f:
        gtLabels = yaml.load(f, Loader=yaml.FullLoader)
    M = gtLabels.get('M')
    R = gtLabels.get('R')
    
    # --- M graphs ---
    fileList = glob.glob("exp_testing/M/M*")
    n = len(fileList)
    if traintest == 'train':
        fileList = fileList[:n//2] #(4*(n//5))
    elif traintest == 'val':
        fileList = fileList[n//2:]
    
    ctr = 0
    for imagePath in fileList:
        correlation = {}
        ctr += 1
        s = imagePath.find('/M/M') + 3
        e = imagePath.find('.', s)
        name = imagePath[s:e]
        print(name)
        if name in M:
            gtCorr = set(M[name].keys())
        else:
            print("key issue in M dict")
            gtCorr = set()
        for corrstr in gtCorr:
            col_corr = corrstr.split()
            correlation[col_corr[0][:-1]] = col_corr[1] # here the slicing on the 0th element is to get rid of the colon at the end of the string
        segImg = segmentImg(imagePath)
        for i,(img,col) in enumerate(segImg):
            closeCol = find_nearest_col(col,posRGB)
            if closeCol in correlation:
                corrstr = correlation[closeCol]
                fname = outputDir + "/" + train_val + "/" + corrstr + "/" + name + '_' + str(i) + ".png" #"seg_" + corrstr + "_" + closeCol + str(i) + "_" + seriesType + "_graph" + str(n) + ".png" # changed to jpg
                plt.imsave(fname,img)
            else:
                print('closest color not found')
                print(col)
                print(closeCol)
                print(correlation)
        plt.close('all')
    
    # --- R graphs ---
    fileList = glob.glob("exp_testing/R/R*")
    n = len(fileList)
    if traintest == 1:
        fileList = fileList[:n/2]
    elif traintest == 2:
        fileList = fileList[n/2:]

    ctr=0
    for imagePath in fileList:
        correlation = {}
        ctr += 1
        s = imagePath.find('/R/R') + 3
        e = imagePath.find('.', s)
        name = imagePath[s:e]
        print(name)
        if name in R:
            gtCorr = set(R[name].keys())
        else:
            print("key issue in R dict")
            gtCorr = set()
        for corrstr in gtCorr:
            col_corr = corrstr.split()
            correlation[col_corr[0][:-1]] = col_corr[1] # here the slicing on the 0th element is to get rid of the colon at the end of the string
        segImg = segmentImg(imagePath)
        for i,(img,col) in enumerate(segImg):
            closeCol = find_nearest_col(col,posRGB)
            if closeCol in correlation:
                corrstr = correlation[closeCol]
                fname = outputDir + "/" + train_val + "/" + corrstr + "/" + name + '_' + str(i) + ".png" #"seg_" + corrstr + "_" + closeCol + str(i) + "_" + seriesType + "_graph" + str(n) + ".png" # changed to jpg
                plt.imsave(fname,img)
            else:
                print('closest color not found')
                print(col)
                print(closeCol)
                print(correlation)
        plt.close('all')

# create_filtered_dirs("wild_graphs_filtered")
# resave_wild_graphs_for_classification("train", "wild_graphs_filtered", traintest="train")
# resave_wild_graphs_for_classification("validation", "wild_graphs_filtered", traintest="val")


# ---------- EXPERIMENTS ----------


# Exp 1
def exp1(numGraphs):
    # generate test graphs
    poss_sNs = [1,2,3]
    ks = []
    pks = []
    sNs = []
    for i in range(numGraphs):
        sN = choice(poss_sNs)
        print(sN)
        sNs.append(sN)
        name = create_multiData(i, sN, 'train', 'random', 'multi', 'multi', 'exp1')
        img = cv2.imread(name)
        ks.append(elbowM(img)-1)
        img = sat_thresh_filter(img,40)
        print('finished preprocessing')
        pks.append(elbowM(img)-1)
    print(sNs)
    print(ks)
    print(pks)
    pscore = 0
    score = 0
    for i in range(len(sNs)):
        if ks[i] == sNs[i]:
            score +=1
        if pks[i] == sNs[i]:
            pscore += 1
    print(score/len(sNs))
    print(pscore/len(sNs))
    nPP = score/len(sNs)
    PP = pscore/len(sNs)


    # plot results
    plt.style.use('default')
    X1 = ['No Preprocessing', 'Preprocessing']
    X2 = [nPP, PP]
    plt.bar(X1, X2)
    plt.ylabel("K Choice Accuracy")
    plt.show()
    #plt.savefig("exp1.png")

#exp1(100)


# Exp 2
def exp2(numGraphs):
    poss_sNs = [1,2,3]
    for i in range(numGraphs):
        sN = choice(poss_sNs)
        print(sN)
        name = create_multiData(i, sN, 'train', 'random', 'multi', 'multi', 'exp2')
        cat = predictCategory(name, "models/correlation/graph_class_model.h5", ['negative', 'neutral', 'positive'])

# Exp 3
def exp3(numGraphs, wild=False):
    # generate test graphs
    poss_sNs = [1,2,3]
    CURRks = []
    CNNks = []
    KNEEDLEks = []
    sNs = []
    for i in range(numGraphs):
        sN = choice(poss_sNs)
        print(sN)
        sNs.append(sN)
        name = create_multiData(i, sN, 'train', 'random', 'multi', 'multi', 'exp1')
        img = cv2.imread(name)
        #img = sat_thresh_filter(img,40)
        print('finished preprocessing')
        CURRks.append(elbowM(img)-1)
        Image.fromarray(img).save('./exp3_placeholder.png')
        CNNks.append(find_k('./exp3_placeholder.png')-1)
        KNEEDLEks.append(elbowM(img, kneedleBasic=True)-1)
    print(sNs)
    print(CURRks)
    print(CNNks)
    print(KNEEDLEks)
    CURRscore = 0
    CNNscore = 0
    KNEEDLEscore = 0
    for i in range(len(sNs)):
        if CURRks[i] == sNs[i]:
            CURRscore += 1
        if CNNks[i] == sNs[i]:
            CNNscore += 1
        if KNEEDLEks[i] == sNs[i]:
            KNEEDLEscore += 1
    CURR = CURRscore/len(sNs)
    CNN = CNNscore/len(sNs)
    KNEEDLE = KNEEDLEscore/len(sNs)
    print(CURR, CNN, KNEEDLE)

    # plot results
    plt.style.use('default')
    X1 = ['Current Method', 'CNN', 'Basic Kneedle Algorithm']
    X2 = [CURR, CNN, KNEEDLE]
    plt.bar(X1, X2)
    plt.ylabel("K Choice Accuracy")
    plt.show()
    #plt.savefig("exp1.png")

#exp3(100)

# Exp 4
'''Do this manually'''

# Exp 5
'''Already Done'''

# Exp 6
'''Already Done'''


# Exp 7
def exp7(numGraphs):
    
    cwd=os.getcwd()
    #clean(cwd)
    


    ### CONTROL ###
    # create encessary dirs
    #outputDir = "exp7_filtered_ctrl"
    #create_filtered_dirs(outputDir)
    # All wild graphs
    #resave_wild_graphs_for_classification("train", outputDir, traintest='train')
    #resave_wild_graphs_for_classification("validation", outputDir, traintest='val')
    # Classifier
    graph_classification(cwd,0,"wild_graphs_filtered")
    

    ### INTERVENTION ###
    # create encessary dirs
    outputDir = "exp7_filtered_test"
    create_filtered_dirs(outputDir)
    # generated graphs
    poss_sNs = [1,2,3]
    for i in range(numGraphs):
        sN = choice(poss_sNs)
        print(sN)
        create_multiData(i, sN, 'train', 'random', 'multi', 'multi', 'g', outputDir=outputDir)
    # wild graphs
    resave_wild_graphs_for_classification("validation", outputDir)
    # Classifier
    graph_classification(cwd,0,outputDir)

#exp7(100)

# Experiment 8
def exp8(numGraphs):
    
    cwd=os.getcwd()

    # Control
    # create necessary dirs
    #clean(os.getcwd())
    outputDir = "exp8_filtered_ctrl"
    create_filtered_dirs(outputDir)
    poss_sNs = [1,2,3]
    for i in range(numGraphs):
        sN = choice(poss_sNs)
        print(sN)
        create_multiData(i, sN, 'train', 'random', 'multi', 'multi', 'g', pstyle='multi', outputDir=outputDir)
    for i in range(numGraphs):
        sN = choice(poss_sNs)
        print(sN)
        create_multiData(i, sN, 'validation', 'random', 'multi', 'multi', 'g', pstyle='multi', outputDir=outputDir)
    # Classifier
    graph_classification(cwd,0,outputDir)
    
    # intervention
    # create necessary dirs
    #clean(os.getcwd())
    outputDir = "exp8_filtered_test"
    create_filtered_dirs(outputDir)
    for i in range(numGraphs):
        sN = choice(poss_sNs)
        print(sN)
        create_multiData(i, sN, 'train', 'random', 'multi', 'multi', 'g', pstyle='default', outputDir=outputDir)
    print("finished part 1")
    for i in range(numGraphs):
        sN = choice(poss_sNs)
        print(sN)
        create_multiData(i, sN, 'validation', 'random', 'multi', 'multi', 'g', pstyle='multi', outputDir=outputDir)
    # Classifier
    graph_classification(cwd,0,outputDir)

#exp8(100)
#cwd = os.getcwd()
#graph_classification(cwd, 0, "exp8_filtered_test_copy")


# Experiment 9)
def exp9(numGraphs):

    # create control directory
    path = "./exp9_ctrl"
    try:
        os.mkdir(path)
    except OSError:
        print ("Warning: Creation of the directory %s failed, might already exist" % path)

    # generate dataset to test OCR
    posSNs = [1,2,3]
    #display_test_dic = {}
    #leg_test_dic = {}
    axis_hits_OD = 0
    axis_hits_algo = 0
    for i in range(numGraphs):
        #leg_display_str = ""
        name,x1s,x2s,tstr,label_to_corr_map = create_multiData(i, choice(posSNs), "train", "random", "multi", "solid", 'pt', outputDir2='exp9_ctrl')
        iname = name + ".png"
        display_string = name + ", x axis: " + x1s + ", y axis: " + x2s + ", title: " + tstr #subsitute: "path + iname" for "name" to match original structure from pipeline_testing.py
        imgPath = path + "/" + iname
        jpgimg = Image.open(imgPath).convert('RGB')
        newimgp = imgPath[:len(imgPath)-3] + 'jpg' # convert png to jpg
        jpgimg.save(newimgp)
        ocr = OCR(imgPath,assign_labels(show_inference(detection_model, newimgp)))
        text_dict_OD = ocr.crop()
        display_string_OD = name
        text_dict_algo = {'x axis': ocr.xAxisLab, 'y axis': ocr.yAxisLab, 'title': ocr.title}
        display_string_algo = name
        #print(text_dict_algo)
        for elem in text_dict_OD:
            if elem != 'legend' and text_dict_OD[elem] is not None:
                display_string_OD = display_string_OD + ", " + elem + ": " + ' '.join(text_dict_OD[elem])
        for elem in text_dict_algo:
            if elem != 'legend' and text_dict_algo[elem] is not None:
                display_string_algo = display_string_algo + ", " + elem + ": " + text_dict_algo[elem]
        
        print(display_string)
        print(display_string_OD)
        print(display_string_algo)

        edds_OD = editfast(display_string, display_string_OD)
        if edds_OD < 6:
            axis_hits_OD = axis_hits_OD + 1
        else:
            print('axis fail OD --------------------------------')
        edds_algo = editfast(display_string, display_string_algo)
        if edds_algo < 6:
            axis_hits_algo = axis_hits_algo + 1
        else:
            print('axis fail algo --------------------------------')
    
    print("OD score: " + str(axis_hits_OD/numGraphs))
    print("algo score: " + str(axis_hits_algo/numGraphs))

    '''
    This is the method for when using both display and legend if only using display use code above
    '''
        # display_test_dic[iname] = display_string
        # leg_set = set()
        # for key in label_to_corr_map:
        #     leg_set.add(key + ": " + label_to_corr_map[key])
        #     leg_display_str = leg_display_str + ", " + key + ": " + label_to_corr_map[key]
        # #ground_truth.append((iname, x1s, x2s, tstr, leg_display_str))
        # leg_test_dic[iname] = leg_display_str

    # data = {"display": display_test_dic, "legend": leg_test_dic}

    # with open('images_labels.yml', 'w') as outfile:
    #     yaml.dump(data, outfile, default_flow_style=False)

    # # test OCR
    # with open('images_labels.yaml') as f:
    #     import_dicts = yaml.load(f, Loader=yaml.FullLoader)
    # displays = import_dicts.get('display')
    # legends = import_dicts.get('legend')
    # fileList = glob.glob("exp9_ctrl/*.png")
    # for imagePath in fileList:
    #     name = imagePath[10:-4]
    #     print(name)

#exp9(100)