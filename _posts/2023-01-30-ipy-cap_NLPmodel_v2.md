---
layout: post
title: "ipy-cap_NLPmodel_v2"
date: 2023-01-30 03:53:22 -0500
categories: GA, NLP
---

The majority of this code is copied directly from the tensorflow website. My real contribution is just in the data wrangling/prep that happened prior to this model-fitting & evaluating


```python
%matplotlib inline

from google.colab import drive
drive.mount('/content/drive')

import os
import shutil

import matplotlib.pyplot as plt
import re
import string
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import losses
```

    Mounted at /content/drive
    


```python
import tarfile

tar_path = 'drive/MyDrive/Capstone'
tar_name = '/cap_reviews.tar.gz'

with tarfile.open(tar_path + tar_name, 'r:gz') as f:
  f.extractall('keras')
```


```python
os.chdir('/content')

main_directory = os.path.abspath('keras/keras')
train_year = '2020'
other_years = os.listdir(main_directory).remove(train_year)
train_dir = os.path.join(main_directory, train_year, "train")

batch_size = 32
seed = 42

raw_train_ds =  tf.keras.utils.text_dataset_from_directory(
    train_dir,
    batch_size = batch_size,
    #validation_split = 0.2,
    #subset = 'training',
    seed = seed
)
```

    Found 25000 files belonging to 2 classes.
    


```python
# this will be altered according to overfit conditions
'''
raw_val_ds = tf.keras.utils.text_dataset_from_directory(
    train_dir,
    batch_size=batch_size,
    validation_split = 0.2,
    subset='validation',
    seed=seed
)'''
```




    "\nraw_val_ds = tf.keras.utils.text_dataset_from_directory(\n    train_dir,\n    batch_size=batch_size,\n    validation_split = 0.2,\n    subset='validation',\n    seed=seed\n)"




```python
# this I need to hold somewhat hostage and come back to
raw_test_dss = {}
other_years = [year for year in os.listdir(main_directory) if year != train_year and year not in ['2008','2009','2010']]

for y in other_years:
  test_dir = os.path.join(main_directory, y, "test")

  raw_test_dss[y] = (tf.keras.utils.text_dataset_from_directory(
      test_dir,
      batch_size=batch_size
  ))
```

    Found 25000 files belonging to 2 classes.
    Found 25000 files belonging to 2 classes.
    Found 25000 files belonging to 2 classes.
    Found 25000 files belonging to 2 classes.
    Found 25000 files belonging to 2 classes.
    Found 25000 files belonging to 2 classes.
    Found 25000 files belonging to 2 classes.
    Found 25000 files belonging to 2 classes.
    Found 25000 files belonging to 2 classes.
    

# Preprocessing


```python
def custom_standardization(input_data):
  lowercase = tf.strings.lower(input_data)
  stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
  return tf.strings.regex_replace(stripped_html,
                                  '[%s]' % re.escape(string.punctuation),
                                  '')
```


```python
max_features = 10_000
sequence_length = 250

vectorize_layer = layers.TextVectorization(
    standardize = custom_standardization,
    max_tokens = max_features,
    output_mode = 'int',
    output_sequence_length = sequence_length
)
```


```python
#make a text-only dataset then call adapt
train_text = raw_train_ds.map(lambda x, y: x)
vectorize_layer.adapt(train_text)
```


```python
def vectorize_text(text, label):
  text = tf.expand_dims(text, -1)
  return vectorize_layer(text), label
```


```python
# now we apply these preprocessing steps

train_ds = raw_train_ds.map(vectorize_text)
#val_ds = raw_val_ds.map(vectorize_text)

test_dss = {}
for year in raw_test_dss.keys():
  test_dss[year] = raw_test_dss[year].map(vectorize_text)
```


```python
# this is a performance-enhancing step. caching allows the data to be stored on-disk in one large file and .prefetch() overlaps preprocessing and model execution

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
#val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
for year in test_dss.keys():
  test_dss[year] = test_dss[year].cache().prefetch(buffer_size=AUTOTUNE)
```

## time for the model


```python
embedding_dim = 16

model = tf.keras.Sequential([
    layers.Embedding(max_features + 1, embedding_dim),
    #layers.Dropout(0.2),
    layers.GlobalAveragePooling1D(),
    #layers.Dropout(0.2),
    layers.Dense(1)]
)

model.summary()
```

    Model: "sequential_18"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     embedding_18 (Embedding)    (None, None, 16)          160016    
                                                                     
     global_average_pooling1d_18  (None, 16)               0         
      (GlobalAveragePooling1D)                                       
                                                                     
     dense_18 (Dense)            (None, 1)                 17        
                                                                     
    =================================================================
    Total params: 160,033
    Trainable params: 160,033
    Non-trainable params: 0
    _________________________________________________________________
    


```python
model.compile(loss=losses.BinaryCrossentropy(from_logits=True),
              optimizer='adam',
              metrics=tf.metrics.BinaryAccuracy(threshold=0.0))
```

#### Training


```python
#training

epochs = 10
history = model.fit(
    train_ds,
    #validation_data=val_ds,
    epochs=epochs
)
```

    Epoch 1/10
    782/782 [==============================] - 7s 8ms/step - loss: 0.6092 - binary_accuracy: 0.6823
    Epoch 2/10
    782/782 [==============================] - 4s 5ms/step - loss: 0.4474 - binary_accuracy: 0.8270
    Epoch 3/10
    782/782 [==============================] - 4s 5ms/step - loss: 0.3346 - binary_accuracy: 0.8934
    Epoch 4/10
    782/782 [==============================] - 4s 5ms/step - loss: 0.2719 - binary_accuracy: 0.9155
    Epoch 5/10
    782/782 [==============================] - 4s 5ms/step - loss: 0.2338 - binary_accuracy: 0.9266
    Epoch 6/10
    782/782 [==============================] - 4s 5ms/step - loss: 0.2086 - binary_accuracy: 0.9339
    Epoch 7/10
    782/782 [==============================] - 4s 5ms/step - loss: 0.1906 - binary_accuracy: 0.9378
    Epoch 8/10
    782/782 [==============================] - 4s 5ms/step - loss: 0.1768 - binary_accuracy: 0.9412
    Epoch 9/10
    782/782 [==============================] - 4s 6ms/step - loss: 0.1657 - binary_accuracy: 0.9445
    Epoch 10/10
    782/782 [==============================] - 4s 5ms/step - loss: 0.1563 - binary_accuracy: 0.9476
    

## Evaluation


```python
#os.chdir('drive/MyDrive/Capstone')

for year in test_dss.keys():
  loss, accuracy = model.evaluate(test_dss[year])

  print(year, ": Loss: ", loss)
  print(year, ": Accuracy: ", accuracy)

  history_dict = history.history

  acc=history_dict['binary_accuracy']
  #val_acc=history_dict['val_binary_accuracy']
  loss=history_dict['loss']
  #val_loss=history_dict['val_loss']

  epochs = range(1, len(acc) + 1)

```

    782/782 [==============================] - 4s 5ms/step - loss: 0.2263 - binary_accuracy: 0.9165
    2018 : Loss:  0.22633349895477295
    2018 : Accuracy:  0.9164800047874451
    782/782 [==============================] - 4s 5ms/step - loss: 0.2770 - binary_accuracy: 0.8926
    2014 : Loss:  0.2769565284252167
    2014 : Accuracy:  0.8925999999046326
    782/782 [==============================] - 4s 5ms/step - loss: 0.2193 - binary_accuracy: 0.9195
    2019 : Loss:  0.2192772626876831
    2019 : Accuracy:  0.9195200204849243
    782/782 [==============================] - 4s 5ms/step - loss: 0.3545 - binary_accuracy: 0.8618
    2012 : Loss:  0.3545071482658386
    2012 : Accuracy:  0.8618000149726868
    782/782 [==============================] - 4s 5ms/step - loss: 0.2530 - binary_accuracy: 0.9056
    2016 : Loss:  0.2530452013015747
    2016 : Accuracy:  0.9056400060653687
    782/782 [==============================] - 4s 5ms/step - loss: 0.2351 - binary_accuracy: 0.9102
    2017 : Loss:  0.23514722287654877
    2017 : Accuracy:  0.9101999998092651
    782/782 [==============================] - 4s 5ms/step - loss: 0.2707 - binary_accuracy: 0.8986
    2015 : Loss:  0.2706727981567383
    2015 : Accuracy:  0.8985999822616577
    782/782 [==============================] - 4s 5ms/step - loss: 0.3101 - binary_accuracy: 0.8799
    2013 : Loss:  0.31008559465408325
    2013 : Accuracy:  0.8798800110816956
    782/782 [==============================] - 4s 6ms/step - loss: 0.3898 - binary_accuracy: 0.8459
    2011 : Loss:  0.38981765508651733
    2011 : Accuracy:  0.8458799719810486
    


```python
 # "bo" is for blue dot
  plt.plot(epochs, loss, 'bo', label='Training Loss')
  # "b" is for solid blue line
  plt.plot(epochs, val_loss, 'b', label='Validation Loss')
  plt.title('Training and Validation Loss')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.legend()

  plt.savefig("med_losses.png")

  plt.plot(epochs, acc, 'bo', label='Training acc')
  plt.plot(epochs, val_acc, 'b', label='Validation acc')
  plt.title('Training and validation accuracy')
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.legend(loc='lower right')

  plt.savefig("med_acc.png")
```


```python
#the history can show us what happened during training (it was recorded by model.fit)


```


```python

```


```python

```

### Now you can export the model


```python
export_model = tf.keras.Sequential([
    vectorize_layer,
    model,
    layers.Activation('sigmoid')
])

export_model.compile(
    loss=losses.BinaryCrossentropy(from_logits=False), optimizer='adam', metrics=['accuracy']
)

#Test it with 'raw_test_ds', which yields raw strings
for year in raw_test_dss.keys():
  loss, accuracy = export_model.evaluate(raw_test_dss[year])
  print(accuracy)
```

    782/782 [==============================] - 13s 16ms/step - loss: 0.3473 - accuracy: 0.8552
    0.855239987373352
    


```python
export_model = tf.keras.Sequential([
    vectorize_layer,
    model,
    layers.Activation('sigmoid')
])

export_model.compile(
    loss=losses.BinaryCrossentropy(from_logits=False), optimizer='adam', metrics=['accuracy']
)

#Test it with 'raw_test_ds', which yields raw strings
for year in raw_test_dss.keys():
  loss, accuracy = export_model.evaluate(raw_test_dss[year])
  print(accuracy)
```


```python
"""
# this stuff is to tell me what variables were set during the course of this script. 

all_variables = dir()

# Iterate over the whole list where dir( )
# is stored.
for name in all_variables:

# Print the item if it doesn't start with '__'
  if not name.startswith('__'):
    myvalue = eval(name)
     print(name, "is", type(myvalue), "and is equal to ", myvalue)

"""
```
