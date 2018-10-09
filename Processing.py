
# coding: utf-8

# In[4]:


import tensorflow as tf
from PIL import Image
import numpy as np
import glob
import os
from sklearn.model_selection import train_test_split


# In[2]:


def listup_files(path):
    return [os.path.abspath(p) for p in glob.glob(path)]


# In[5]:


tclist = listup_files('/Users/kenjimatsumoto1983/ResNetProject/train/TC/*.tif')
nontclist = listup_files('/Users/kenjimatsumoto1983/ResNetProject/train/nonTC/*.tif')


# In[ ]:


# x（画像のバイナリ配列）の配列初期化（３次元配列）
x = np.empty((0,64,64), np.float32)
# y（正否ラベル）の配列初期化（３次元配列）
y = np.empty((0,1), int)

# 正解データを配列化
for filepath in tclist:
    image = Image.open(filepath)
    item = np.array(image)
    x = np.append(x, np.array([item]), axis=0)
    y = np.append(y, np.array([[1]]), axis=0)

# 不正解データを配列化
count = 0
for flipath in nontclist:
    image = Image.open(filepath)
    x = np.append(x, np.array([item]), axis=0)
    y = np.append(y, np.array([[0]]), axis=0)
    count += 1
    print(count)


# In[16]:


print(x.shape)
print(y.shape)


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1)

