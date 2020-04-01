# coding: utf-8

# In[1]:


import os
import cv2
import numpy as np

# In[2]:

path = r"C:\Users\FIRAT-PC\Desktop\CROP\BRN_ELEME"

os.chdir(path)

os.getcwd()

dosya = os.listdir(path)

boyut = 512
sayi = len(dosya)

# In[9]:

images = np.zeros((sayi, boyut, boyut, 3), dtype=np.uint16)

i = -1
for data in dosya:
    i = i + 1
    im = cv2.imread(data, -1)
    np_im = np.asarray(im)
    images[i, :, :, :] = np_im

np_images = np.asarray(images)
np.save('train_BRN', np_images)
