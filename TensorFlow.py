#!/usr/bin/env python
# coding: utf-8

# In[8]:


get_ipython().system('pip install numpy pandas tensorflow tensorflow_hub scikit-learn openpyxl')
get_ipython().system('pip install --upgrade tensorflow tensorflow_hub')


# In[15]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense

# Step 1: Load the data
data = pd.read_excel("Medical_data.xlsx")

# Step 2: Preprocess the data
X_text = data['Gender'].values
y = data['Condition'].values

# Step 3: Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Step 4: Tokenize text data
max_words = 1000  # Maximum number of words to keep
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_text)
X_seq = tokenizer.texts_to_sequences(X_text)

# Pad sequences to ensure uniform length
max_len = max(len(seq) for seq in X_seq)
X_pad = pad_sequences(X_seq, maxlen=max_len)

# Step 5: Build the neural network model
embedding_dim = 50
model = Sequential()
model.add(Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_len))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step 6: Train the model
X_train, X_test, y_train, y_test = train_test_split(X_pad, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Step 7: Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy)


# In[16]:


model.summary()


# In[27]:


data.head(25)


# In[25]:


data.shape


# In[26]:


data.info()


# In[29]:


data.tail(15)


# In[30]:


data.columns


# In[32]:


data.isnull().sum()


# In[33]:


data.duplicated().sum


# In[35]:


data['Condition'].value_counts()

