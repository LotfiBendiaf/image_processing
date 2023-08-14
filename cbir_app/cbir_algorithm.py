import numpy as np
from numpy import linalg as LA
from PIL import Image

from keras.applications.resnet import ResNet50
from keras.applications.resnet import preprocess_input

class ResNetNet:
    def __init__(self):
        self.input_shape = (224, 224, 3)
        self.weight = 'imagenet'
        self.pooling = 'max'
        self.model = ResNet50(weights=self.weight, input_shape=self.input_shape, pooling=self.pooling, include_top=False)
        self.model.predict(np.zeros((1, 224, 224, 3)))

    def extract_feat(self, img_path):
        img = Image.open(img_path).resize((self.input_shape[0], self.input_shape[1]))
        img = np.array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        feat = self.model.predict(img)
        norm_feat = feat[0] / LA.norm(feat[0])
        return norm_feat
