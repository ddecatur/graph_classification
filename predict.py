import cv2
import tensorflow as tf

#CATEGORIES = [1,2,3]
#CATEGORIES = ['positive', 'negative', 'neutral']

def prepare(filepath):
    IMG_SIZE = 150
    img_array = cv2.imread(filepath)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3)

def load_model(fname):
    return tf.keras.models.load_model(fname)

def predict(img, model):
    prediction = model.predict([prepare(img)])
    return prediction

def predictCategory(img='./testttttt.png',model='series_class_model_v1.h5',CATEGORIES=[1,2,3]):
    prediction = predict(img, load_model(model))
    # model = load_model(model)
    # newpred = model.predict_classes([prepare(img)]) #model.predict_classes(prediction)
    # newpred=newpred[0]
    midx = 0
    mval = 0
    for i,pred in enumerate(prediction[0]):
        if pred > mval:
            midx = i
            mval = pred
    if mval == 0:
        print("error: no category found")
        return "[error: no category found]"
    else:
        #print("seriesnum: ")
        #print(CATEGORIES[midx])
        #print(img)
        return CATEGORIES[midx]#CATEGORIES[newpred]CATEGORIES[midx]
#predictCategory('test3.png')
#print(predictCategory('test3.png','graph_class_model.h5',['positive', 'negative', 'neutral']))