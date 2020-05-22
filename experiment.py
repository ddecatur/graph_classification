from create_graph import *
from graph_classification import graph_classification
from series_classification import series_classification
from clean import *
from process_instF import *
from seg_img import *
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
seg = instr.get('seg')
sN = instr.get('seriesNum')
ST = instr.get('seriesTrain')

i=0
for i in range(0,n): # run the classiciation model n times for the given color setup
    executeOrder66(args.instF,cwd) # create the training data from the instructions file
    # if seg:
    #     saveGraphs()
    #     cleanReg(cwd)
    if ST:
        series_classification(cwd, i)
        cleanSeries(cwd)
    else:
        graph_classification(cwd, i)
        clean(cwd)