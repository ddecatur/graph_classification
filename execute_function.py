# Note: call with a command line argument of the desired number of graphs to generate
# import necessary files
from create_graph import *
from graph_classification import *
from clean import *
import sys
import csv
import argparse

# parse command line arguments
parser = argparse.ArgumentParser(description='Parse Command Line Arguments')
parser.add_argument('function', help='which function you want to execute')
parser.add_argument('-size', help='how many graphs to generatate')
parser.add_argument('-train', action='store_true', help='if multi color training data, by default single color')
parser.add_argument('-val', action='store_true', help='if multi color validation data, by default single color')
parser.add_argument('-v', action='store_true')
parser.add_argument('-gcPath', help='path to directory containing folder of filtered graphs (usually current directory)')
parser.add_argument('-cleanPath', help='path to directory that you want to run the clean function from (usually current directory)')
args = parser.parse_args()

# process input variables
if(args.function == "creating_training_data"):
    create_data(int(args.size), args.train, args.val, args.v, os.getcwd())
if(args.function == "graph_classification"):
    graph_classification(args.gcPath,1)
if(args.function == "clean"):
    clean(args.cleanPath)
else:
    print("error: invalid function specified")