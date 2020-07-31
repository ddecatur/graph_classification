from predict import *
from seg_img import *
from create_graph import find_nearest_col
from PIL import Image
import pytesseract
from ocr import *
from object_prediction import *

posRGB = {(255,0,0):'r', (0,128,0):'g', (0,0,255):'b', (255,165,0):'o', (128,128,128):'gr'}
colorMap = {'r':'red', 'g':'green', 'b':'blue', 'o':'orange', 'gr':'gray'}


def avg_height(img):
    return 0

def run(img):
    
    #create new dir
    path = "./pipeline_batch"
    try:
        os.mkdir(path)
    except OSError:
        print ("Warning: Creation of the directory %s failed, might already exist" % path)
    
    
    rtn = []
    col_to_cat_map = {}
    segImg = segmentImg(img)
    ocr = OCR(img,segImg,assign_labels(show_inference(detection_model, new_img_path)))
    text_dict = ocr.crop()
    if ocr.match_leg_img:
        ocr.match_leg_img.save("legend_cropped.jpg")
        #segLeg = segmentImg("legend_cropped.jpg")
    print(text_dict)
    print('segimg len:')
    print(len(segImg))
    for i,(res,col) in enumerate(segImg):
        fname = "pipeline_batch/" + str(i) + ".png"
        plt.imsave(fname, res)
        cat = predictCategory(fname, "models/correlation/graph_class_model_v3.h5", ['negative', 'neutral', 'positive'])
        # variable = pytesseract.image_to_string(Image.open(fname))
        col = find_nearest_col(col,posRGB)
        '''
        for each segmented thing, find box closest to an exisintg pixel
        '''
        col_to_cat_map[col] = cat
        # if col in rtn:
        #     col=col+str(i)
        #     colorMap[col]=col
        #     print('pipeline error: overwrite color')
        # rtn[variable] = cat
        rtn.append(cat)
    # color_list = []
    # for res,col in segLeg:
    #     color_list.append(avg_height(res))
    # color_list.sort(reverse=True)
    
    return (rtn,text_dict)

#results = list()
# for i in range(0,3):
#     results.append(run('./test' + str(i) + '.png'))
# for result in results:
#     print(result)
#new_img_path = './images_test/graph_0.jpg'
new_img_path = './images/test.jpg'
result,text_dict = run(new_img_path)
for cat in result:
    display_string = cat + " correlation"
    for elem in text_dict:
        display_string = display_string + ", " + elem + ": "
        for obj in text_dict[elem]:
            display_string = display_string + " " + obj
    print(display_string)

# for elem in result:
#     col = colorMap.get(elem)
#     print(col)
#     print("color " + col + " has a " + result.get(elem) + " correlation")