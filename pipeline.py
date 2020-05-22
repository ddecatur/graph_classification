from predict import *
from seg_img import *
from create_graph import find_nearest_col

posRGB = {(255,0,0):'r', (0,255,0):'g', (0,0,255):'b'}
colorMap = {'r':'red', 'g':'green', 'b':'blue'}


def run(img):
    
    #create new dir
    path = "./pipeline_batch"
    try:
        os.mkdir(path)
    except OSError:
        print ("Warning: Creation of the directory %s failed, might already exist" % path)
    
    
    rtn = {}
    segImg = segmentImg(img)
    for i,(res,col) in enumerate(segImg):
        fname = "pipeline_batch/" + str(i) + ".png"
        plt.imsave(fname, res)
        cat = predictCategory(fname, "graph_class_model.h5", ['positive', 'negative', 'neutral'])
        col = find_nearest_col(col,posRGB)
        rtn[col] = cat
    return rtn


result = run('./testttttt.png')
for elem in result:
    col = colorMap.get(elem)
    print("color " + col + " has a " + result.get(elem) + " correlation")