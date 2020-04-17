from create_graph import *
import yaml

def executeOrder66(instrF, directory):
    # determine what type of graph to build
    # read in yaml instructions file
    with open(instrF) as f:
        instr = yaml.load(f, Loader=yaml.FullLoader)

    # turn the strings from yaml into the correct type
    (t1,t2) = instr.get('graphType').split(',')
    (c1,c2) = instr.get('color').split(',')
    (s1,s2) = instr.get('dataStyle').split(',')
    bools = [c1,c2,s1,s2]
    for Bool in bools:
        if Bool=='True':
            Bool=True
        else:
            Bool=False

    # create the training data
    create_training_data(instr.get('size'), (t1,t2), (c1,c2), (s1,s2), instr.get('verbose'), directory)