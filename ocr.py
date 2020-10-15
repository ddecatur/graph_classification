import cv2
import numpy as np
import math
import datetime
import pytesseract
import re
from pytesseract import Output
from seg_img import segmentImg
from PIL import Image
from object_prediction import *
import matplotlib.pyplot as plt
import re

class OCR():

    def __init__(self, image, segImg, box_dict, k): # , rotation=):
        self.img = cv2.imread(image)
        #self.img = self.img*0.5
        self.img = cv2.resize(self.img, (0,0), fx=3, fy=3)
        #self.img = cv2.GaussianBlur(self.img,(11,11),0)
        #self.img = cv2.medianBlur(self.img,9)
        print('save')
        im = Image.fromarray(self.img)
        im.save("images/test.png")
        #self.img = cv2.GaussianBlur(self.img,(11,11),0)
        #self.img = cv2.medianBlur(self.img,9)
        self.image_obj = Image.open(image)
        self.get_grayscale()
        #self.di = []
        self.box_dict = box_dict
        self.match_leg_img = None
        self.leg_box = None
        self.leg_text_boxes = {}
        self.crop_amount = 35
        self.k = k
        self.xAxisLab = None
        self.yAxisLab = None
        self.title = None
        self.legend = []
        # self.remove_noise()
        # self.thresholding()
        #self.mser()
        #print(self.img[0][0])
        #print(self.img[10][5])
        self.dimensions = self.img.shape
        #self.LEFT_THRESHOLD = self.dimensions[1] / 5
        #self.BOTTOM_THRESHOLD = self.dimensions[0] / 5
        #self.seg = {}
        #for res,col in segImg:
            #self.seg[tuple(col)] = res
        self.d = pytesseract.image_to_data(self.img, config='--psm 3 -c tessedit_char_whitelist=0123456789-_abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ', output_type=Output.DICT)
        self.get_boxes()
        #self.seriesCorsp = {}
    
    def isREGEX(self, string):
        regex = re.compile('[.@=»_!#$%^&*()<>?/\|}{~:]') # adding hyphons to prevent dashed legend lines from confusing the ocr
        return (regex.search(string) == None)

    def remove_key(self, box):
        (xmin, ymin, xmax, ymax) = box
        return (xmin+self.crop_amount, ymin, xmax, ymax)

    def crop(self):
        text_dict = {}
        self.match_leg_img = None
        for box in self.box_dict:
            crp_img = self.image_obj
            if box == None:
               self.box_dict[box] = 'extra text'
            crp_img = crp_img.crop(box)
            if self.box_dict[box] == 'y axis':
                crp_img = crp_img.rotate(270, expand=True) # might need to be more general in the future
                #crp_img.show()
            if self.box_dict[box] == 'legend':
                #(xmin, ymin, xmax, ymax) = box
                #newbox = (xmin-50, ymin, xmax, ymax)
                leg_crop = self.image_obj
                leg_crop = leg_crop.crop(self.remove_key(box))
                #crp_img = self.image_obj.crop(newbox)
                #crp_img.show()
                #leg_crop.show()
                self.match_leg_img = crp_img
                self.leg_box = box
                crp_img = leg_crop
            # break
            crpD = pytesseract.image_to_data(crp_img, config='--psm 3 -c tessedit_char_whitelist=0123456789-_abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ', output_type=Output.DICT)
            #crpD = pytesseract.image_to_data(crp_img, output_type=Output.DICT) # note change back to crp_img
            n_boxes = len(crpD['text'])
            strlist = []
            # for elem in crpD['text']:
            #     if (not elem.isspace()) and elem!='' and self.isREGEX(elem) and elem!=',':
            #         strlist.append(elem)#st + " " + elem
            for i in range(0,n_boxes):
                elem = crpD['text'][i]
                if (not elem.isspace()) and elem!='' and elem!=',' and elem!='-' and elem!='—' and elem!='_': #and self.isREGEX(elem) 
                    strlist.append(elem)
                    if self.box_dict[box] == 'legend':
                        self.leg_text_boxes[elem] = (crpD['left'][i] + (crpD['width'][i])/2, crpD['top'][i] + (crpD['height'][i])/2) # 

            text_dict[self.box_dict[box]] = strlist

            # if self.box_dict[box] == 'legend':
            #     for token in st.split():
                    
        return text_dict

    def dist(self,p1,p2):
        return math.sqrt(((p1[0]-p2[0])**2) + ((p1[1]-p2[1])**2))

    def get_boxes(self):
        print(self.dimensions)
        #xMin, yMax = self.dimensions[1], 0
        maxY = self.dimensions[0]
        xMaxDP, yMaxDP, yMaxDP2 = 0, 0, 0
        xMidPos = self.dimensions[1]//2
        yMidPos = self.dimensions[0]//2
        yMaxDPIdx = None
        yMaxDPIdx2 = None
        xMaxDPIdx = None
        maxYIdx = None
        minDPIdx = None
        minDP = max(self.dimensions[0],self.dimensions[1])
        n_boxes = len(self.d['text'])
        print(self.d['text'])
        #print(self.d['top'])
        for i in range(n_boxes):
            elem = self.d['text'][i]
            if (not elem.isspace()) and elem!='' and elem!=',' and elem!='-' and elem!='—' and elem!='_': #int(self.d['conf'][i]) > 60 and 
                #print((elem, self.d['top'][i]))
                (x,y) = (self.d['left'][i] + (self.d['width'][i])/2, self.d['top'][i] + (self.d['height'][i])/2)
                if abs(x-xMidPos) >= xMaxDP:
                    xMaxDP = abs(x-xMidPos)
                    xMaxDPIdx = i
                if y < maxY: # because reverse coordinates
                    maxY = y
                    maxYIdx = i
                if abs(y-yMidPos) >= yMaxDP:
                    yMaxDP2 = yMaxDP
                    yMaxDPIdx2 = yMaxDPIdx
                    yMaxDP = abs(y-yMidPos)
                    yMaxDPIdx = i
                elif abs(y-yMidPos) >= yMaxDP2:
                    yMaxDP2 = abs(y-yMidPos)
                    yMaxDPIdx2 = i
                dst = self.dist((x,y), (xMidPos,yMidPos))
                if dst < minDP:
                    minDP = dst
                    minDPIdx = i
        if xMaxDPIdx is not None:
            self.yAxisLab = self.d['text'][xMaxDPIdx]
        if yMaxDPIdx is not None:
            if yMaxDPIdx != maxYIdx:
                self.xAxisLab = self.d['text'][yMaxDPIdx]
            elif yMaxDPIdx2 is not None:
                self.xAxisLab = self.d['text'][yMaxDPIdx2]
        if maxYIdx is not None:
            self.title = self.d['text'][maxYIdx]
        #print((self.yAxisLab,self.title))
        #self.legend.append(self.d['text'][minDPIdx])
        #print('legend:')
        #print(self.legend)
        # for i in range(n_boxes):
        #     while len(self.legend < 3):


    def mser(self):
        '''
        METHOD #1
        '''
        _, bw = cv2.threshold(self.img, 0.0, 255.0, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
        connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
        contours, hierarchy,=cv2.findContours(connected.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        counter=0
        array_of_texts=[]
        for idx in range(len(contours)):
            x, y, w, h = cv2.boundingRect(contours[idx])
            cropped_image = self.image_obj.crop((x-10, y, x+w+10, y+h ))
            str_store = re.sub(r'([^\s\w]|_)+', '', pytesseract.image_to_string(cropped_image))
            d = pytesseract.image_to_data(cropped_image, config='--psm 12 -c tessedit_char_whitelist=0123456789.abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ', output_type=Output.DICT)#pytesseract.image_to_string(cropped_image))
            self.di.append(d)
            array_of_texts.append(str_store)
            counter+=1
        self.idvdilen = len(contours)
        #print(array_of_texts)
        #print('len of cont: ' + str(self.idvdilen))
        
        
        self.di.append({'text': ['now on to method #2']})
        '''
        METHOD #2
        '''
        # mser = cv2.MSER_create()
        # vis = self.img.copy()
        # regions, bboxes = mser.detectRegions(self.img)
        # hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
        # cv2.polylines(vis, hulls, 1, (0, 255, 0))
        # # cv2.imshow('img', vis)
        # # cv2.waitKey(0)
        # mask = np.zeros((self.img.shape[0], self.img.shape[1], 1), dtype=np.uint8)
        # for contour in hulls:
        #     x, y, w, h = cv2.boundingRect(contours[idx])
        #     cropped_image = self.image_obj.crop((x-10, y, x+w+10, y+h ))
        #     cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)
        #     d = pytesseract.image_to_data(cropped_image, config='--psm 12', output_type=Output.DICT)#pytesseract.image_to_string(cropped_image))
        #     self.di.append(d)
        # text_only = cv2.bitwise_and(self.img, self.img, mask=mask)
        # # cv2.imshow("text only", text_only)
        # # cv2.waitKey(0)


        # vis = self.img.copy()
        # mser = cv2.MSER_create()
        # regions = mser.detectRegions(self.img)
        # hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
        # cv2.polylines(vis, hulls, 1, (0, 255, 0))
        # cv2.imshow('img', vis)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # for bbox in bboxes:
        #     (x,y,w,h) = tuple(bbox)
        #     # print(bbox)
        #     display = cv2.rectangle(self.img, (x,y), (x+w, y+h), (0,255,0), 2)
        # print('regions')
        # for reg in regions:
        #     # print(reg)
        #     print(pytesseract.image_to_string(reg))

        # cv2.imshow('display', display)
        # cv2.waitKey(0)

    # convert to grayscale image
    def get_grayscale(self):
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
    
    # noise removal
    def remove_noise(self):
        self.img = cv2.GaussianBlur(self.img,(11,11),0)

    #thresholding
    def thresholding(self):
        self.img = cv2.threshold(self.img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    def axisLab(self):
        xMin, yMax = self.dimensions[1], 0
        for i in range(self.n_boxes):
            if int(self.d['conf'][i]) > 60 and (not self.d['text'][i].isspace()):
                #print(self.d['text'][i])
                (x,y) = (self.d['left'][i], self.d['top'][i])
                if x <= xMin:
                    xMin = x
                    xMinIdx = i
                if y >= yMax:
                    yMax = y
                    yMaxIdx = i
        if (not xMinIdx or not yMaxIdx):
            print('error: ocr -- min or max not set')
        if xMinIdx == yMaxIdx:
            print('x and y label are same text')
            self.xAxisLab = self.d['text'][xMinIdx]
            self.yAxisLab = self.d['text'][yMaxIdx]
        else:
            self.xAxisLab = self.d['text'][xMinIdx]
            self.yAxisLab = self.d['text'][yMaxIdx]
    

    def bbDist(self): # O(n*m)
        for i in range(0, self.n_boxes):
            if int(self.d['conf'][i]) > 60 and (not self.d['text'][i].isspace()) and self.d['text'][i]!='':
                print(self.d['text'][i])
                minD = self.dimensions[0] + self.dimensions[1]
                pt = (self.d['left'][i], self.d['top'][i])
                coordinates = {}
                currCol = [-1]
                for col in self.seg:
                    indices = np.where(self.seg[col] != [0])
                    coordinates[col] = zip(indices[0], indices[1])
                    for cord in coordinates[col]:
                        dist = self.dist(cord, pt)
                        if dist <= minD:
                            minD = dist
                            currCord = cord
                            currCol = col
                if currCol!=[-1]:
                    self.seriesCorsp[currCol] = self.d['text'][i]
                else:
                    print('never found min distance')

# start = datetime.datetime.now()
# segImg = segmentImg('images/test.jpg') # 'test_images/test1.png') #legend_test.png')
# print('segImg elapsed time: ', (datetime.datetime.now()-start).total_seconds())
# start = datetime.datetime.now()
# label_dict = assign_labels(show_inference(detection_model, single_img_path))
# ocr = OCR('images/test.jpg', segImg, label_dict) # 'test_images/test1.png', segImg) #legend_test.png', segImg)
# res = ocr.crop()
# print(res)
# # for i in range(ocr.n_boxes):
# #     if int(ocr.d['conf'][i]) > 60:
# #         (x,y,w,h) = (ocr.d['left'][i], ocr.d['top'][i], ocr.d['width'][i], ocr.d['height'][i])
# #         img = cv2.rectangle(ocr.img, (x, y), (x + w, y + h), (0, 255, 0), 2)
# # # cv2.imshow('img', img)
# # # cv2.waitKey(0)
# # ocr.axisLab()
# print('OCR elapsed time: ', (datetime.datetime.now()-start).total_seconds())
# #print(ocr.d['text'])
# print(ocr.xAxisLab)
# print(ocr.yAxisLab)
# ocr.bbDist()
# print(ocr.seriesCorsp)
# for i,elem in enumerate(ocr.di):
#     #print(elem)
#     #print('new dict')
#     for text in elem['text']:
#         #print(text)
#         if (not text.isspace()) and text!='':
#             print(text)
# for i in range(0,ocr.idvdilen):
#     print(ocr.di[i]['text'])
# ocr.mser()

# # ------------ Following preprocessing functions taken from https://nanonets.com/blog/ocr-with-tesseract/ ------------


# # noise removal
# def remove_noise(image):
#     return cv2.medianBlur(image,5)
 


# #dilation
# def dilate(image):
#     kernel = np.ones((5,5),np.uint8)
#     return cv2.dilate(image, kernel, iterations = 1)
    
# #erosion
# def erode(image):
#     kernel = np.ones((5,5),np.uint8)
#     return cv2.erode(image, kernel, iterations = 1)

# #opening - erosion followed by dilation
# def opening(image):
#     kernel = np.ones((5,5),np.uint8)
#     return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

# #canny edge detection
# def canny(image):
#     return cv2.Canny(image, 100, 200)

# #skew correction
# def deskew(image):
#     coords = np.column_stack(np.where(image > 0))
#     angle = cv2.minAreaRect(coords)[-1]
#     if angle < -45:
#         angle = -(90 + angle)
#     else:
#         angle = -angle
#     (h, w) = image.shape[:2]
#     center = (w // 2, h // 2)
#     M = cv2.getRotationMatrix2D(center, angle, 1.0)
#     rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
#     return rotated

# #template matching
# def match_template(image, template):
#     return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
# # ------------ End preprocessing functions ------------