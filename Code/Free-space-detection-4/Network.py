import cv2
import numpy as np
import json
import tensorflow as tf
from tensorflow import keras


#import the trained network
#model = tf.keras.models.load_model("../RAW_data/IDX/model.h5")
model = tf.keras.models.load_model("./RAW_data/IDX/model.h5")
probability_model = tf.keras.Sequential([model,tf.keras.layers.Softmax()])

def getimgfromjson(img):
    # path_ofjson = "../RAW_data/josn_file_foto/c5001.json"
    path_ofjson = "./RAW_data/josn_file_foto/c5001.json"
    #open josn file
    with open(path_ofjson, 'r') as f:
        locobj = json.load(f)

    #get point from josn
    dir = locobj['shapes']
    length = len(dir)
    # print(length)
    empty = []
    full = []

    for i in range(length):

        if (dir[i]['label'] == 'car'):

            #four points in josn
            topleft = dir[i]['points'][0]
            bottomleft = dir[i]['points'][1]
            topright = dir[i]['points'][3]
            bottomright = dir[i]['points'][2]

            pts1 = np.float32([topleft, topright, bottomleft, bottomright])

            #change image format to network
            pts = np.array([topleft,bottomleft,bottomright,topright], np.int32)
            pts = pts.reshape((-1, 1, 2))
            pts2 = np.float32([[0, 0], [28, 0], [0, 28], [28, 28]])
            M = cv2.getPerspectiveTransform(pts1, pts2)

            #get rectangle from josn
            dst = cv2.warpPerspective(img, M, (28, 28))

            #transform image to gary (28*28) mnist
            dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
            dst = cv2.resize(dst, (28, 28))
            dst = np.reshape(dst,(-1))
            dst = (np.expand_dims(dst,0))
            predictions = probability_model.predict(dst)

            #make a desicision
            if (np.argmax(predictions) == 1):
                empty.append(pts)
            else:
                full.append(pts)

    return empty,full