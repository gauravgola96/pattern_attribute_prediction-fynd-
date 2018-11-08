import warnings
warnings.filterwarnings('ignore')

import os,cv2
import keras
from PIL import Image
import pandas as pd
import numpy as np
from keras.models import Model
from keras.layers import Dropout,GlobalAveragePooling2D,Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model,Sequential
from keras.layers import Dense,Flatten
from keras.layers import BatchNormalization,Activation

import argparse





# Loading model with no weights
print('--------------LOADING MODEL NETWORK-------------------')
model_Xcep = keras.applications.xception.Xception(weights=None,include_top=False,input_shape=(299,299,3))

# Customizing network for our dataset(12 classes)
x = model_Xcep.output
x = Dropout(0.5)(x)
x = GlobalAveragePooling2D()(x)
x = Dense(512,activation='sigmoid')(x)
x = Dropout(0.5)(x)
pred = Dense(12,activation='softmax')(x)
model = Model(inputs=model_Xcep.input, outputs=pred)



# Best epoch model
print('-------------LOADING MODEL WEIGHTS----------')
model.load_weights('/home/gaurav/Documents/GAURAV/Test/AML/final/model_checkpoint/patterntest.08-1.69.hdf5')



pattern_dict = {0: 'Checked',1: 'Colourblock',2: 'Melange',3: 'Patterned',4: 'Printed',5: 'abstract',6: 'floral',7: 'graphic',8: 'polka dots',9: 'solid',
 10: 'striped',11: 'typography'}



print('-------------LOADING REQUIRED UDFs----------------')
def crop_image(input_):
    im = Image.open(input_)
    crop_rectangle = (50, 150, 550, 600)
    cropped_im = im.crop(crop_rectangle)
    return cropped_im


def image_preprocess(image):
    #cropping
    img = crop_image(image)
    #Read image
#     img = cv2.imread(image)
    img = np.array(img)
    img.resize(1,299,299,3)
    return img
    
    
def predict(image):
    image_processed = image_preprocess(image)
    pred = model.predict(image_processed)
    y = np.argmax(pred,axis=1)
    prob = np.max(pred)
    return (pattern_dict[y[0]]),prob


def main(image):
    image_processed = image_preprocess(image)
    pred = model.predict(image_processed)
    y = np.argmax(pred,axis=1)
    prob = np.max(pred)
    return print(pattern_dict[y[0]])
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Run pre-trained model on a single image')
    parser.add_argument('--image_path',metavar='path',help='Enter image path',required=True)
    args = parser.parse_args()
    
    
    main(image=args.image_path)

