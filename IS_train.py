"""
This is the Imagined Speech test set - CNN model

When calling the models.py, the model may be different to VI as the input is differently shaped.
"""

from methodis import *
import os
import sys
from typing import List, Tuple
import mne
# sys.path.append("/workspace")
import numpy as np
import tensorflow as tf
import pickle
import scipy.io
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import minmax_scale
tf.autograph.set_verbosity(0)
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print(physical_devices)
# config = tf.config.experimental.set_memory_growth(physical_devices, True)
from models import HopefullNet, HopefullNet_HBN


### IMAGINED SPEECH

# 1. load MNE preprocessed epochs for each participant (nExamples*nChannels*nTime)

os.getcwd()

is_epochs = mne.read_epochs('/Users/dianele/Desktop/Data/braincontrol_ml/notebooks/diane_time_frequency/is_epoch_15-epo.fif')

is_data = is_epochs.get_data()


# y array
is_labels = is_epochs.events[:, -1]  # -1 last column where all labels are


# 2. Apply feature selection

# 3. Apply data augumentation techinque (random cropped training)
# Generate augmented data - Random Cropped training

is_data, is_labels = generate_segmented_epochs(X=is_data,
                                               y=is_labels,
                                               srate=256,
                                               overlap_factor= 0.9,
                                               segment_length= 0.5)

#print(is_data.shape)
#sys.exit()


# data, labels = generate_segmented_data() -> nAugExamples*nChannels*nTime,
# e.g. 300*45*128, where 128 is 0.5s (3 examples from 1, with same label)

is_data = np.reshape(is_data,(is_data.shape[0], is_data.shape[-1], is_data.shape[1]))


# Separate out the training set (80), validation set(10) and test set(10)
X_train, X_test, y_train, y_test = train_test_split(is_data,
                                                    is_labels,
                                                    stratify=is_labels,
                                                    test_size=0.10,
                                                    random_state=1)

X_train, X_val, y_train, y_val = train_test_split(X_train,
                                                  y_train,
                                                  stratify=y_train,
                                                  test_size=0.10,
                                                  random_state=1)

##(nExamples*nChannels*nTime)


train_set = (X_train, y_train)
val_set = (X_val, y_val)
test_set = (X_test, y_test)

train_set,val_set,test_set = convert_labels(train_set, val_set, test_set, [4,5])

# remove from tuple
X_train, y_train = train_set
X_val, y_val = val_set
X_test, y_test = test_set

# To apply one-hot-encoding
y_train = to_one_hot(y_train)
y_val = to_one_hot(y_val)
y_test = to_one_hot(y_test)

# Tuple
train_set = (X_train, y_train)
val_set = (X_val, y_val)
test_set = (X_test, y_test)


# Reshape the model to reflect the dimension: (nExamples*nTime*nChannels)

input_shape = (None, is_data.shape[1], is_data.shape[-1])



# 4. Instantiate the CNN model
## TRAINING ##

learning_rate = 1e-4

loss = tf.keras.losses.binary_crossentropy
optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
model = HopefullNet(inp_shape=input_shape)
model.build(input_shape)
modelPath = os.path.join(os.getcwd(), 'braincontrol_ml', 'notebooks', 'diane_time_frequency', "models", "test")

model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

checkpoint = ModelCheckpoint( # set model saving checkpoints
    modelPath, # set path to save model weights
    monitor='val_loss', # set monitor metrics
    verbose=1, # set training verbosity
    save_best_only=True, # set if want to save only best weights
    save_weights_only=False, # set if you want to save only model weights
    mode='auto', # set if save min or max in metrics
    period=1 # interval between checkpoints
    )

earlystopping = EarlyStopping(
    monitor='val_loss', # set monitor metrics
    min_delta=0.001, # set minimum metrics delta
    patience=20, # number of epochs to stop training
    restore_best_weights=True, # set if use best weights or last weights
    )
callbacksList = [checkpoint, earlystopping] # build callbacks list

hist = model.fit(X_train, y_train, epochs=100, batch_size=10,
                validation_data=(X_val, y_val), callbacks=callbacksList) #32

loss = hist.history['loss']
acc = hist.history['accuracy']
val_loss = hist.history['val_loss']
val_acc = hist.history['val_accuracy']

print()
print(f'train_loss: {loss}')
print(f'val_loss: {val_loss}')
print(f'accuracy: {acc}')
print(f'val_accuracy: {val_acc}')

# SAVE TRAINED MODEL on training dataset
# model.save_weights(checkpoint_path.format(epoch=0))

save_path = os.path.join(os.getcwd(), 'braincontrol_ml', 'notebooks', 'diane_time_frequency', "models", "test")
if not os.path.exists(save_path):
    os.mkdir(save_path)
with open(os.path.join(save_path, "hist.pkl"), "wb") as file:
    pickle.dump(hist.history, file)


# TESTING THE MODEL
del model # Delete the original model, just to be sure!
model = tf.keras.models.load_model(save_path)
testLoss, testAcc = model.evaluate(X_test, y_test)
print('\nAccuracy:', testAcc)
print('\nLoss: ', testLoss)

from sklearn.metrics import classification_report, confusion_matrix
yPred = model.predict(X_test)

# convert from one hot encode in string
yTestClass = np.argmax(y_test, axis=1)
yPredClass = np.argmax(yPred, axis=1)

print('\n Classification report \n\n',
  classification_report(
      yTestClass,
      yPredClass,
       target_names=["Relax", "Push"]
      )
  )

print('\n Confusion matrix \n\n',
  confusion_matrix(
      yTestClass,
      yPredClass,
      )
  )

# Plot training and validation loss
#plt.figure(figsize=(10, 6))
#plt.plot(loss, label='Train Loss', color='blue')
#plt.plot(val_loss, label='Validation Loss', color='orange')
#plt.xlabel('Epoch')
#plt.ylabel('Loss')
#plt.title('Loss')
#plt.legend()#
#plt.grid()
#plt.show()