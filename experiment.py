from create_graph import *
from graph_classification import graph_classification
from series_classification import series_classification
from clean import *
from process_instF import *
from seg_img import *
import argparse
import datetime

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
GP = instr.get('genPurpose')

i=0
for i in range(0,n): # run the classiciation model n times for the given color setup
    start = datetime.datetime.now()
    executeOrder66(args.instF,cwd) # create the training data from the instructions file
    if GP == 'series_train':
        series_classification(cwd, i)
        cleanSeries(cwd)
    else:
        graph_classification(cwd, i)
        clean(cwd)
    print('generation and classification ' + str(i) + ' elapsed time: ', (datetime.datetime.now()-start).total_seconds())