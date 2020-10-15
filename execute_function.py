# Note: call with a command line argument of the desired number of graphs to generate
# import necessary files
from create_graph import *
from graph_classification import *
from series_classification import *
from clean import *
from process_instF import *
from seg_img import *
import sys
import csv
import argparse
import os
import yaml

# parse command line arguments
parser = argparse.ArgumentParser(description='Parse Command Line Arguments')
parser.add_argument('function', help='which function you want to execute')
parser.add_argument('-instF', default='instructions.yaml',
    help='instructions on the parameters for data generation; see README for info')
args = parser.parse_args()
print("function argument: " + args.function)

# determine cwd
cwd = os.getcwd()

# process input variables
if(args.function == "create_training_data"):
    print("successfully recognized args.function as create_training_data")
    # execute order 66 aka generate the training data
    executeOrder66(args.instF, cwd)
elif(args.function == "graph_classification"):
    graph_classification(cwd,1)
elif(args.function == "series_classification"):
    series_classification(cwd,1)
elif(args.function == "clean"):
    clean(cwd)
elif(args.function == "cleanSeries"):
    cleanSeries(cwd)
else:
    print("error: invalid function specified")