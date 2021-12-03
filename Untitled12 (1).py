#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df1 = pd.read_csv("C:\\spotify_dataset.csv")


# In[3]:


pd.set_option("display.max.columns",None)
df1


# In[4]:


df1["artist"].nunique()


# In[5]:


df1.drop(df1[["track","uri","decade"]],axis = 1,inplace = True)
df1


# In[6]:


df1.isnull().sum()


# In[7]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df1["artist"]=le.fit_transform(df1["artist"])


# In[8]:


df1


# In[9]:


X = df1.drop('popularity',axis=1)
X


# In[10]:


y = df1["popularity"]


# In[11]:


df1.corr()["popularity"]


# In[12]:


from sklearn.preprocessing import MinMaxScaler
mm = MinMaxScaler()


# In[13]:


col = X.columns
col


# In[14]:


X = pd.DataFrame(mm.fit_transform(X),columns=col)
X


# In[15]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=1)
X_train,y_train,X_test,y_test


# In[16]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()


# In[17]:


lr.fit(X_train,y_train)


# In[18]:


y_pred = lr.predict(X_test)


# In[19]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_pred)


# In[20]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state=0)
rf.fit(X_train,y_train)


# In[21]:


y_pred = rf.predict(X_test)


# In[22]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_pred)


# In[23]:


from sklearn.svm import SVC
svc = SVC(kernel = "poly", C = 10)
svc.fit(X_train,y_train)


# In[24]:


y_pred = svc.predict(X_test)


# In[25]:


confusion_matrix(y_test,y_pred)


# In[40]:


from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
model = Sequential([
    Dense(128,activation='relu',input_shape=[16]),
    Dropout(0.3),
    Dense(64,activation='relu'),
    Dropout(0.3),
    Dense(32,activation='relu'),
    Dropout(0.3),
    Dense(1,'sigmoid')
])


# In[41]:


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'],)


# In[42]:


import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(min_delta=0.0001,patience=20,restore_best_weights=True)
history = model.fit(X_train,y_train,epochs=500,batch_size=2000,callbacks=[early_stopping],validation_data=(X_test,y_test))


# In[43]:


history_df = pd.DataFrame(history.history)

history_df.loc[:, ['loss', 'val_loss']].plot();
print("Minimum validation loss: {}".format(history_df['val_loss'].min()))


# In[44]:


y_pred = model.predict(X_test)


# In[45]:


y_pred = [1 if i>0.5 else 0 for i in y_pred]


# In[46]:


y_pred


# In[47]:


confusion_matrix(y_test,y_pred)


# In[34]:


from sklearn.neural_network import MLPClassifier
mlp=MLPClassifier(random_state=1, max_iter=10000,batch_size=2000)
mlp.fit(X_train,y_train)


# In[35]:


y_pred = mlp.predict(X_test)
y_pred


# In[36]:


confusion_matrix(y_test,y_pred)

