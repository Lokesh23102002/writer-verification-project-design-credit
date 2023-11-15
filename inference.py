from utility import TestUtils
import config
import tensorflow as tf
import numpy as np

def prediction2(model,path1,path2):
    proba = []
    y_preds = []
    ids = []
    notx = 0
    img1 = []
    index = []
    
    generated_frag = TestUtils.fragment_generator(path1,path2)


    y_pred =  model.predict([generated_frag[:,0],generated_frag[:,1]])
    return np.mean(y_pred)


model2 = tf.keras.models.load_model("D:/Writer_handwriting_classifier/vgg16_model/archive/vgg16_nn_data.h5")
model2.summary()

print(
    prediction2(model2,"D:/Writer_handwriting_classifier/dataset/dataset/train/P172/A1.jpg","D:/Writer_handwriting_classifier/dataset/dataset/train/P172/B0.jpg"))