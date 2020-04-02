import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
from keras.models import load_model

l = os.listdir("../data/images")
image = Image.open(f'../data/images/{l[3]}')
image = image.convert('L')
image = np.array(image)
print(len(image))
print(len(image[0]))
# plt.imshow(image, cmap='gray)
# image = image[1000:2000]
n = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)[1]
thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)[1]
plt.imshow(n, cmap='gray')
plt.show()
cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(len(cnts))
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
pi = []
posx = []
posy = []
size = []
for c in cnts:
    x, y, w, h = cv2.boundingRect(c)
    if w + h > 50:
        posx.append(x)
        posy.append(y)
        size.append([w, h])
        print(x, y, w, h)
        pi.append(n[y:y + h, x:x + w])
posx = np.array(posx)
pi2 = posx.argsort()
pi = np.array(pi)
p3 = pi[pi2]
p = p3.copy()
for i in range(len(p)):
    if p[i].shape[0] > p[i].shape[1]:
        zero = np.full((p[i].shape[0], int((p[i].shape[0] - p[i].shape[1]) / 2)), 255)
        p[i] = np.concatenate((zero, p[i]), axis=1)
        p[i] = np.concatenate((p[i], zero), axis=1)
    else:
        zero = np.full((int((p[i].shape[1] - p[i].shape[0]) / 2), p[i].shape[1]), 255)
        p[i] = np.concatenate((zero, p[i]), axis=0)
        p[i] = np.concatenate((p[i], zero), axis=0)
res = []
for ps in p:
    res.append(cv2.resize(ps, dsize=(45, 45), interpolation=cv2.INTER_NEAREST))
m = load_model("../data/models/third_model.h5")
for i in range(len(res)):
    res[i] = res[i] / 255
res = np.array(res)
res = res.reshape(len(res), 45, 45, 1)
pred = m.predict_classes(res)
dic = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'plus': 10, 'minus': 11,
       'X': 12, 'y': 13, 'z': 14, 'comma': 15, 'pareno': 16, 'parenc': 17}
predicted = [0 for i in range(len(pred))]
for i in range(len(pred)):
    predicted[i] = list(dic.keys())[pred[i]]
dic2 = {'0': '0', '1': '1', '2': '2', '3': '3', '4': '4', '5': '5', '6': '6', '7': '7', '8': '8', '9': '9', 'plus': '+',
        'minus': '-', 'X': 'x', 'y': 'y', 'z': 'z', 'comma': ',', 'pareno': '(', 'parenc': ')'}
a = [dic2[i] for i in predicted]
s = ''
for i in a:
    s += i
s = s.replace('--', '=')
print(s)
