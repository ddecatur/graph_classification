from create_graph import *
from graph_classification import graph_classification
from clean import clean
import OpenSSL
import argparse

# parse command line arguments
parser = argparse.ArgumentParser(description='Parse Command Line Arguments')
parser.add_argument('-gNum', default=1, help='how many graphs to generatate')
parser.add_argument('-lNum', default=1, help='how many iterations')
parser.add_argument('-train', action='store_true', help='if multi color training data, by default single color')
parser.add_argument('-val', action='store_true', help='if multi color validation data, by default single color')
parser.add_argument('-v', action='store_true', help='verbose mode')
args = parser.parse_args()

n = int(args.lNum)
cwd=os.getcwd()
i=0
for i in range(0,n): # run the classiciation model n times for the given color setup
    create_data(int(args.gNum), args.train, args.val, args.v, cwd)
    graph_classification(cwd, i)
    clean(cwd)
    print("cool")