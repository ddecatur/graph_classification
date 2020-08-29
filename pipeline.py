from predict import *
from seg_img import *
from create_graph import find_nearest_col
from PIL import Image
import pytesseract
from ocr import *
from object_prediction import *
from numpy.random import random
import math
import csv # dont need anymore
import glob
import cv2

posRGB = {(255,0,0):'r', (0,128,0):'g', (0,0,255):'b', (255,165,0):'o', (128,128,128):'gr'}
colorMap = {'r':'red', 'g':'green', 'b':'blue', 'o':'orange', 'gr':'gray'}

def dist(p1,p2):
    return math.sqrt(((p1[0]-p2[0])**2) + ((p1[1]-p2[1])**2))

def permutation(lst):
    if len(lst) == 0: 
        return [] 
  
    if len(lst) == 1: 
        return [lst] 
  
    l = [] # empty list that will store current permutation 
  
    # Iterate the input(lst) and calculate the permutation 
    for i in range(len(lst)): 
       m = lst[i] 
  
       # Extract lst[i] or m from the list.  remLst is 
       # remaining list 
       remLst = lst[:i] + lst[i+1:] 
  
       # Generating all permutations where m is first 
       # element 
       for p in permutation(remLst): 
           l.append([m] + p) 
    return l 


def match_series(col_to_seg_map,offset,leg_text_boxes,img_shape):
    '''
    returns a dictionary associating colors to series text
    '''
    # print(img_shape.shape[0])
    # print(img_shape.shape[1])
    print('leg_text_boxes: ')
    print(leg_text_boxes)
    rtn = {}
    specialph = []
    matching = {}
    dist_list = []
    combin = {}
    texts = []
    colors = []
    for elem in leg_text_boxes:
        texts.append(elem)
        currCol = None
        (x,y) = leg_text_boxes[elem]
        pt = (x + offset, y)
        #print(pt)
        minD = img_shape.shape[0] + img_shape.shape[1]
        colors = []
        for col in col_to_seg_map:
            coordinates = []
            xcords = []
            ycords = []
            img = col_to_seg_map[col]
            # if col == 'bdfjhdjfhd':
            #     for col in img:
            #         for row in col:
            #             specialph.append(row)
            #     #indices = np.where(img != [0], [255], [0])
            #     #mask_img = Image.fromarray(indices, 'RGB')
            #     mask = np.array(indices, dtype=np.uint8)

            #     cv2.imshow('mask', mask)
            #     cv2.waitKey()
            #     #mask_img.show()
            # else:
            #     #indices = np.where(img != [0])
            #     aaa = 2
            for i in range(0,img_shape.shape[0]): # here using a manual loop through since np.where not working
                for j in range(0,img_shape.shape[1]):
                    #for rgb in img[i][j]:
                    if img[i][j] != 0:
                        # if rgb != 0:
                        #     xcords.append(j)
                        #     ycords.append(i)
                        #     coordinates.append((j,i))
                        #     break
                        xcords.append(j)
                        ycords.append(i)
                        coordinates.append((j,i))
            #print(len(coordinates))
            if xcords != [] and ycords != []:
                colors.append(col)
                xcords.sort()
                ycords.sort()
                # print("len xcords")
                # print(len(xcords))
                xmed = xcords[int(len(xcords)/2)-1]
                ymed = ycords[int(len(ycords)/2)-1]
                #print(col)
                #print((xmed,ymed))
                distance = dist((xmed,ymed), pt)
                #print(distance)
                if distance in matching:
                    print("warning: same distance")
                    distance = distance + (random()*0.00000001) # hopefully suffficiently small
                dist_list.append(distance)
                #print(dist_list)
                #print('first time')
                #print(elem+col)
                combin[elem+col] = distance
                #combin.append((elem,col,dist))
                matching[distance] = (elem, col)
                # if distance < minD:
                #     minD = distance
                #     currCol = col
                # if col == 'b':
                #     print(distance)
                #     print((xmed,ymed))
                #     print(pt)
                #     return 1
            # print("after done color")
            # print(col)
            # print(minD)
            #print(indices)
            #break
            # print(len(indices[0]))
            # print(len(indices[1]))
            # coordinates = zip(indices[0], indices[1])
            #len(coordinates)
            # for cord in coordinates:
            #     distance = dist(cord, pt)
            #     if distance < minD:
            #         minD = distance
            #         currCol = col
            #         if col == 'b':
            #             print(distance)
            #             print(cord)
            #             print(pt)
            #             return 1
            # print("after done color")
            # print(col)
            # print(minD)
    
        # if currCol!=None:
        #     rtn[currCol] = elem
        # else:
        #     print('never found min distance')
        #print(currCol)
        #print(rtn[currCol])
    #print(dist_list)
    dist_list.sort()
    #print(dist_list)
    #print(matching)
    seen_already = []
    #print(dist_list)


    # ---------------------- OLD ALGO ---------------------
    # for dist_elem in dist_list:
    #     (text, color) = matching[dist_elem]
    #     if text not in seen_already and color not in seen_already:
    #         seen_already.append(text)
    #         seen_already.append(color)
    #         rtn[color] = text
    
    min_perm_dist = -1 # needs to sufficiently large
    final_perm = []
    print(texts)
    print(colors)
    for perm in permutation(colors):
        perm_dist = 0
        #print(len(texts))
        #print(perm)
        #print(len(perm))
        for i in range(0,min(len(texts),len(perm))):
            #print('second time')
            #print(texts[i]+perm[i])
            perm_dist = perm_dist + combin[texts[i]+perm[i]]
        if min_perm_dist == -1 or (perm_dist < min_perm_dist):
            min_perm_dist = perm_dist
            final_perm = perm
    
    for i in range(0,len(final_perm)):
        if len(texts) >= i+1:
            rtn[final_perm[i]] = texts[i]
    #else:
        #rtn[final_perm[i]] = "unknown text"

    print(rtn)
    return rtn


def run(img):
    
    #create new dir
    path = "./pipeline_batch"
    try:
        os.mkdir(path)
    except OSError:
        print ("Warning: Creation of the directory %s failed, might already exist" % path)
    rtn = {}
    col_to_cat_map = {}
    col_to_seg_map = {}
    segImg = segmentImg(img)
    segLeg = None
    ocr = OCR(img,segImg,assign_labels(show_inference(detection_model, img)))
    text_dict = ocr.crop()
    #print(ocr.match_leg_img)
    #if ocr.match_leg_img:
        #print('yes')

        #ocr.match_leg_img.save("legend_cropped.jpg")
        #segLeg = segmentImg("legend_cropped.jpg",fixed_k=len(segImg))
    # print(text_dict)
    # print('segimg len:')
    # print(len(segImg))
    #color_list = []
    image_holder = []
    print("seg len:" + str(len(segImg)))
    for i,(res,col) in enumerate(segImg):
        fname = "pipeline_batch/" + str(i) + ".jpg"
        plt.imsave(fname, res)
        cat = predictCategory(fname, "models/correlation/graph_class_model_v3.h5", ['negative', 'neutral', 'positive'])
        # variable = pytesseract.image_to_string(Image.open(fname))
        col = find_nearest_col(col,posRGB)
        '''
        for each segmented thing, find box closest to an exisintg pixel
        '''
        if ocr.leg_box != None:
            newimg = Image.open(fname)
            crp_res = newimg.crop(ocr.leg_box)
            crp_arr = np.asarray(crp_res)
            crp_gray = cv2.cvtColor(crp_arr, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(crp_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            # threshimg = Image.fromarray(thresh)
            # threshimg.show()
            #thresharr = np.asaray(threshimg)
            col_to_seg_map[col] = thresh#np.asarray(crp_res)
            img_shape = thresh#np.asarray(crp_res)
            
            # for res,col in segImg:
            # self.seg[tuple(col)] = res
            # color_list.append(match_series(np.asarray(crp_res), ocr.crop_amount, ocr.leg_text_boxes)) # added the crop amount here to be able to recover the coordinates of the text boxes
        
        col_to_cat_map[col] = cat
    if ocr.leg_box != None:
        col_to_series_map = match_series(col_to_seg_map, ocr.crop_amount, ocr.leg_text_boxes, img_shape)
        for key in col_to_series_map:
            rtn[col_to_series_map[key]] = col_to_cat_map[key]
    else:
        rtn = col_to_cat_map

        # if col == 'g':
        #     for col in img_shape:
        #         for row in col:
        #             image_holder.append(row)

        #     # write csv files
        #     with open('./image_holder.csv', 'w') as f:
        #         writer = csv.writer(f, delimiter=',')
        #         writer.writerows(image_holder)
        #     f.close()

        # if col in rtn:
        #     col=col+str(i)
        #     colorMap[col]=col
        #     print('pipeline error: overwrite color')
        # rtn[variable] = cat
        #rtn.append(cat)

    
    
    # print('col_to_series_map')
    # print(col_to_series_map)
    

    # color_list = []
    # if segLeg!=None:
    #     print("yes2")
    #     for res,col in segLeg:
    #         fname = "pipeline_batch/test_leg.png"
    #         plt.imsave(fname, res)
    #         color_list.append(avg_height(res))
    # color_list.sort(reverse=True)
    # print(color_list)
    print(col_to_cat_map)
    return (rtn,text_dict)


def process_img(img_path):
    result,text_dict = run(img_path)
    display_string = img_path
    for elem in text_dict:
        if elem != 'legend':
            display_string = display_string + ", " + elem + ":"
            for obj in text_dict[elem]:
                display_string = display_string + " " + obj
    corr_set = set()
    for series in result:
        corr_set.add(series + ": " + result[series])
        #display_string = display_string + ", " + series + ": " + result[series]
    
    return (display_string, corr_set)

strrrr, setttt = process_img("images/OI_5.jpg")#'images_test/test.jpg'))
print(strrrr)
print(setttt)