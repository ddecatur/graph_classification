from create_graph import *
from graph_classification import graph_classification
from clean import clean
from process_instF import *
import OpenSSL
import argparse

# parse command line arguments
parser = argparse.ArgumentParser(description='Parse Command Line Arguments')
parser.add_argument('-instF', default='instructions.yaml', help='yaml file with data generation instructions')
args = parser.parse_args()

cwd=os.getcwd()

with open(args.instF) as f:
    instr = yaml.load(f, Loader=yaml.FullLoader)

n = instr.get('learnNum')

i=0
for i in range(0,n): # run the classiciation model n times for the given color setup
    executeOrder66(args.instF,cwd) # create the training data from the instructions file
    graph_classification(cwd, i)
    clean(cwd)