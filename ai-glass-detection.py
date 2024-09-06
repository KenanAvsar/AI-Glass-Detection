#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random


# In[2]:


import warnings
warnings.filterwarnings('ignore')


# In[3]:


train = pd.read_csv('/kaggle/input/applications-of-deep-learning-wustl-fall-2023/train.csv')


# In[4]:


test = pd.read_csv('/kaggle/input/applications-of-deep-learning-wustl-fall-2023/test.csv')


# In[5]:


train.head()


# In[6]:


train.shape,test.shape


# In[7]:


test.head()


# In[8]:


img = cv2.imread('/kaggle/input/applications-of-deep-learning-wustl-fall-2023/10005.jpg')
plt.imshow(img)


# In[9]:


from PIL import Image
img = Image.open('/kaggle/input/applications-of-deep-learning-wustl-fall-2023/10005.jpg')
#img = img.resize((256,256)) görselin boyutunu ayarlamak için
img


# In[10]:


img = cv2.imread('/kaggle/input/applications-of-deep-learning-wustl-fall-2023/10005.jpg')
img


# In[11]:


img_path = '/kaggle/input/applications-of-deep-learning-wustl-fall-2023'


# In[12]:


random_files = train.sample(16)
random_files['file']


# In[13]:


random_files['file'].iloc[0]


# In[14]:


files = os.listdir(img_path)
random_files = random.sample(files,16)

# random_files görselleştirme
plt.figure(figsize=(10,10))
for i in range(16):
    img = cv2.imread(os.path.join(img_path,random_files[i]))
    plt.subplot(4,4,i+1)
    plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    plt.axis('off')
plt.tight_layout()
plt.show()


# In[15]:


def pre_img(df, img_path, size=128):
    X=[]
    for img in df:
        img_file = os.path.join(img_path,str(img))
        img = cv2.imread(img_file)
        img = cv2.resize(img,(size,size))
        img = img/255.0
        X.append(img)
    X=np.array(X)
    return X


# In[16]:


X_train = pre_img(train['file'],img_path)
y_train = train['glasses']


# In[17]:


from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout,BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, mean_squared_error


# In[18]:


size=128

model = Sequential()

model.add(Conv2D(64,(3,3), activation='relu', input_shape=(size,size,3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(128,(3,3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer = 'adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Erken Durdurma
early_stopping = EarlyStopping(monitor='val_loss',patience=5,verbose=1)


# In[19]:


from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


# In[20]:


history = model.fit(X_train, y_train, validation_data=(X_val,y_val), epochs=50, batch_size=32, callbacks=[early_stopping], verbose=1)


# In[21]:


submission=pd.read_csv('/kaggle/input/applications-of-deep-learning-wustl-fall-2023/sample_submission.csv')
submission.head()


# In[22]:


X_test=pre_img(test['file'],img_path)
submission['glasses']=model.predict(X_test).round().astype(int)
submission.to_csv('submission.csv',index=False)
submission.head()


# In[23]:


model.save("model_glass.keras")

