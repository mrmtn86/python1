# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 01:52:06 2018

@author: firat
"""


# coding: utf-8

# In[1]:


import os
import cv2
import numpy as np


# In[2]:

path = r"C:\Users\FIRAT-PC\Desktop\CROP\BINARY_ELEME"

os.chdir(path)

os.getcwd()

dosya = os.listdir(path)

boyut = 512
sayi = len(dosya)

# In[9]:

images = np.zeros((sayi,boyut,boyut,2),dtype=np.uint8)

i = -1
for data in dosya:
    i = i + 1
    im=cv2.imread(data,0)
    np_im=np.asarray(im)
    
    T = np_im == 255
    F = np_im == 0
    images[i,:,:,0] = T
    images[i,:,:,1] = F
    
np_images=np.asarray(images)
np.save('label',np_images)