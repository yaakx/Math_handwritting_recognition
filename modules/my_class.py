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
        self.contour = self.get_contours()

    def get_contours(self):
        plt.imshow(self.white)
        plt.show()
        cnts = cv2.findContours(self.black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        posx = []
        posy = []
        width = []
        height = []
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            if w + h > 20 and x != 0 and x + w != len(self.black[0]) \
                    and y != 0 and y + h != len(self.black):
                posx.append(x)
                posy.append(y)
                width.append(w)
                height.append(h)
        self.posx = np.array(posx)
        self.posy = np.array(posy)
        self.width = np.array(width)
        self.height = np.array(height)
        size_filter = (self.height + self.width) > max(self.height + self.width) / 10
        self.posx = self.posx[size_filter]
        self.posy = self.posy[size_filter]
        self.width = self.width[size_filter]
        self.height = self.height[size_filter]
        maximumy = self.posy[(self.height + self.width) == max(self.height + self.width)][0]
        maximumh = maximumy + (self.height[(self.height + self.width) == max(self.height + self.width)])[0]
        margin = max(self.height + self.width)
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
        contour = []
        for i in range(len(self.posy)):
            contour.append(self.white[self.posy[i]:self.posy[i] + self.height[i],
                           self.posx[i]:self.posx[i] + self.width[i]])
        return contour

    def swap(self, first_pos, last_pos, pos, up, down):
        self.posx = list(self.posx)
        self.posy = list(self.posy)
        self.width = list(self.width)
        self.height = list(self.height)
        self.posx = np.array(self.posx[0:first_pos] + [self.posx[i] for i in up] + [self.posx[pos]] +
                             [self.posx[i] for i in down] + self.posx[last_pos + 1:])
        self.posy = np.array(self.posy[0:first_pos] + [self.posy[i] for i in up] + [self.posy[pos]] +
                             [self.posy[i] for i in down] + self.posy[last_pos + 1:])
        self.height = np.array(self.height[0:first_pos] + [self.height[i] for i in up] + [self.height[pos]] +
                               [self.height[i] for i in down] + self.height[last_pos + 1:])
        self.width = np.array(self.width[0:first_pos] + [self.width[i] for i in up] + [self.width[pos]] +
                              [self.width[i] for i in down] + self.width[last_pos + 1:])


class Equation(object):
    dic = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
           10: '+', 11: '-', 12: 'x', 13: 'y', 14: 'z', 15: ',', 16: '(', 17: ')'}

    def __init__(self, image, model="../data/models/third_model.h5"):
        self.img = Image.open(image)
        self.image = np.array(self.img.convert('L'))
        if self.image.shape[0] > self.image.shape[1]:
            self.image = np.rot90(self.image)
        self.model = load_model(model)
        self.res = None
        self.numbers = None
        self.predict = None
        self.final_eq = ""
        self.division = []
        self.equation = self.get_equation()

    def get_equation(self):
        if self.predict:
            return self.equation
        self.numbers = Contour(self.image)
        contour = self.numbers.contour
        for i in range(len(contour)):
            if contour[i].shape[0] > contour[i].shape[1]:
                zero = np.full((contour[i].shape[0], int((contour[i].shape[0] - contour[i].shape[1]) / 2)), 255)
                contour[i] = np.concatenate((zero, contour[i]), axis=1)
                contour[i] = np.concatenate((contour[i], zero), axis=1)
            else:
                zero = np.full((int((contour[i].shape[1] - contour[i].shape[0]) / 2), contour[i].shape[1]), 255)
                contour[i] = np.concatenate((zero, contour[i]), axis=0)
                contour[i] = np.concatenate((contour[i], zero), axis=0)
        res = []
        for cont in contour:
            res.append((cv2.resize(cont, dsize=(45, 45), interpolation=cv2.INTER_NEAREST)) / 255)
        res = np.array(res)
        self.res = res.reshape((len(res), 45, 45, 1))
        self.predict = [self.dic[k] for k in self.model.predict_classes(self.res)]
        for i in range(len(self.predict)):
            if self.predict[i] == '-':
                self.minus_verification(i)
        for i in range(len(self.predict)):
            self.verification(i)
        if '=' not in self.predict:
            for i in range(len(self.predict)):
                if self.predict[i] in "yz" or self.predict[i] == "**z" or self.predict[i] == "**y":
                    if "**" in self.predict[i]:
                        self.predict[i] = f'**{self.get_different_pred(i)}'
                    else:
                        self.predict[i] = self.get_different_pred(i)

    def minus_verification(self, pos):
        up = []
        down = []
        for i in (range(len(self.predict))):
            if i == pos:
                pass
            if (self.numbers.posx[pos] < self.numbers.posx[i] < self.numbers.posx[pos] + self.numbers.width[pos]) or \
                    (self.numbers.posx[pos] < self.numbers.posx[i] + self.numbers.width[i] < self.numbers.posx[pos]
                     + self.numbers.width[pos]):
                if self.numbers.posy[i] < self.numbers.posy[pos]:
                    up.append(i)
                else:
                    down.append(i)
        if up or down:
            if (len(up) == 1 and self.predict[up[0]] == '-') or (len(down) == 1 and self.predict[down[0]] == '-'):
                self.predict[pos] = '='
                self.predict[up[0] if up else down[0]] = '='
            elif up and down:
                self.predict[pos] = '/'
                first_pos = min([min(up), min(down), pos])
                last_pos = max([max(up), max(down), pos])
                self.predict = self.predict[0:first_pos] + [self.predict[i] for i in up] + [self.predict[pos]] + \
                               [self.predict[i] for i in down] + self.predict[last_pos + 1:]
                self.res = list(self.res)
                self.res = self.res[0:first_pos] + [self.res[i] for i in up] + [self.res[pos]] + \
                               [self.res[i] for i in down] + self.res[last_pos + 1:]
                self.res = np.array(self.res)
                self.numbers.swap(first_pos, last_pos, pos, up, down)
                self.division.append([i for i in range(first_pos, last_pos + 1)])

    def verification(self, pos):
        if self.predict[pos] != "=" and pos != 0 and self.predict[pos - 1] != '-':
            for i in self.division:
                if pos in i:
                    j = i[0]
                    while '/' not in self.predict[j]:
                        j += 1
                    if '**' in self.predict[j]:
                        return
                    pos = i[0]
                    if pos == 0:
                        return
                    if self.predict[pos - 1] == '-':
                        return
                    not_pos = pos - 1
                    while "**" in self.predict[not_pos]:
                        not_pos -= 1
                    if self.numbers.posy[j + 1] + self.numbers.height[j + 1] < self.numbers.posy[not_pos] + \
                            (self.numbers.height[not_pos] / 1.75):
                        for k in i:
                            self.predict[k] = f'**{self.predict[k]}'
                        return
            not_pos = pos - 1
            while "**" in self.predict[not_pos]:
                not_pos -= 1
            for i in self.division:
                if not_pos > max(i):
                    break
                if not_pos in i:
                    while self.predict[not_pos] != '/':
                        not_pos -= 1
            next_pos = pos + 1
            for i in self.division:
                if next_pos in i:
                    while self.predict[next_pos] != '/':
                        next_pos += 1
                    next_pos += 1
            if next_pos >= len(self.predict):
                next_pos = len(self.predict) - 1
            if self.predict[pos] == '-':
                if self.numbers.posy[next_pos] + self.numbers.height[next_pos] < self.numbers.posy[not_pos] + \
                        (self.numbers.height[not_pos] / 2):
                    self.predict[pos] = f'**{self.predict[pos]}'
            elif self.numbers.posy[pos] + self.numbers.height[pos] < self.numbers.posy[not_pos] + \
                    (self.numbers.height[not_pos] / 2):
                self.predict[pos] = f'**{self.predict[pos]}'

    def get_different_pred(self, pos, pred=2):
        prob = self.model.predict(self.res[pos].reshape((1, 45, 45, 1)))[0]
        for i in range(1, pred + 1):
            max_prob = max(prob)
            for j in range(len(prob)):
                if prob[j] == max_prob:
                    if i != pred:
                        prob[j] = 0
                    else:
                        return self.dic[j]


import os

l = os.listdir("../data/images")
print(l)
d = Equation(f'../data/images/{l[7]}')
print(d.predict)
