# Note: call with a command line argument of the desired number of graphs to generate
# import necessary files
from create_graph import *
from graph_classification import *
from clean import *
import sys
import csv
import argparse
import os

# parse command line arguments
parser = argparse.ArgumentParser(description='Parse Command Line Arguments')
parser.add_argument('function', help='which function you want to execute')
parser.add_argument('-size', help='how many graphs to generatate')
parser.add_argument('-train', action='store_true', help='if multi color training data, by default single color')
parser.add_argument('-val', action='store_true', help='if multi color validation data, by default single color')
parser.add_argument('-line', action='store_true', help='if line graph style, by default scatter')
parser.add_argument('-lineType', action='store_true', help='if multi styled line data, by default solid style')
parser.add_argument('-v', action='store_true')
#parser.add_argument('-gcPath', help='path to directory containing folder of filtered graphs (usually current directory)')
#parser.add_argument('-cleanPath', help='path to directory that you want to run the clean function from (usually current directory)')
args = parser.parse_args()
print("args funct:" + args.function)
# determine cwd
cwd = os.getcwd()

# determine what type of graph to build
if args.line:
    graphType = 'line'
else:
    graphType = 'scatter'

# process input variables
if(args.function == "create_training_data"):
    print("successfully recognized args.function")
    create_data(int(args.size), args.train, args.val, graphType, args.lineType, args.v, cwd)
elif(args.function == "graph_classification"):
    graph_classification(cwd,1)
elif(args.function == "clean"):
    clean(cwd)
else:
    print("error: invalid function specified")