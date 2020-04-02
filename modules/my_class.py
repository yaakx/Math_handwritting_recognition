from keras.models import load_model
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt


class Contour(object):
    def __init__(self, image):
        self.white = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)[1]
        self.black = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)[1]
        self.posx = None
        self.posy = None
        self.width = None
        self.height = None
        self.contour = None

    def get_contours(self):
        cnts = cv2.findContours(self.black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        posx = []
        posy = []
        width = []
        height = []
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            if w + h > 20 and w / h < 10 and h / w < 10 and x != 0 and x + w != len(image[0]) \
                    and y != 0 and y + h != len(image):
                posx.append(x)
                posy.append(y)
                width.append(w)
                height.append(h)
        self.posx = np.array(posx)
        self.posy = np.array(posy)
        self.width = np.array(width)
        self.height = np.array(height)
        filter = (self.height + self.width) > max(self.height + self.width)/10
        self.posx = self.posx[filter]
        self.posy = self.posy[filter]
        self.width = self.width[filter]
        self.height = self.height[filter]
        maximumy = self.posy[(self.height + self.width) == max(self.height + self.width)][0]
        maximumh = maximumy + (self.height[(self.height + self.width) == max(self.height + self.width)])[0]
        margin = (self.height[(self.height + self.width) == max(self.height + self.width)])[0] / 2
        i = 0
        while i < len(self.posy):
            if i >= len(self.posy):
                break
            if (self.posy[i] < maximumy - margin and self.posy[i] + self.height[i] < maximumy - margin) or \
                    (self.posy[i] > maximumh + margin and self.posy[i] + self.height[i] > maximumh + margin):
                self.posx = np.delete(self.posx, i)
                self.posy = np.delete(self.posy, i)
                self.height = np.delete(self.height, i)
                self.width = np.delete(self.width, i)
                i -= 1
            i += 1
        position = self.posx.argsort()
        self.posx = self.posx[position]
        self.posy = self.posy[position]
        self.height = self.height[position]
        self.width = self.width[position]
        self.contour = []
        for i in range(len(self.posy)):
            self.contour.append(self.white[self.posy[i]:self.posy[i] + self.height[i],
                                self.posx[i]:self.posx[i] + self.width[i]])


class Equation(object):
    dic = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4',
           5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
           10: '+', 11: '-', 12: 'X',
           13: 'y', 14: 'z', 15: ',',
           16: '(', 17: ')'}

    def __init__(self, image, model="../data/models/third_model.h5"):
        self.img = Image.open(image)
        self.image = np.array(self.img.convert('L'))
        self.model = load_model(model)
        self.numbers = None
        self.predict = None
        self.equation = self.get_equation()

    def get_equation(self):
        if self.predict:
            return self.equation


import os

l = os.listdir("../data/images")
img = Image.open(f'../data/images/{l[4]}')
image = np.array(img.convert('L'))
d = Contour(image)
d.get_contours()
for i in d.contour:
    plt.imshow(i)
    plt.show()
plt.imshow(image)
plt.show()
print(len(d.posx))
print(d.posx)
print(d.posy)
print(d.height)
print(d.width)

