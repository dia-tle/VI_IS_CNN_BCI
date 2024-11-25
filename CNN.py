import os
import sys
# sys.path.append("/workspace")
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import pickle
from sklearn.preprocessing import minmax_scale

tf.autograph.set_verbosity(0)
physical_devices = tf.config.experimental.list_physical_devices('GPU')
# print(physical_devices)
# config = tf.config.experimental.set_memory_growth(physical_devices, True)
from models import HopefullNet, HopefullNet_HBN


# generate augmented data - Cropped training
def generate_segmented_epochs(X: np.ndarray,
                              y: np.ndarray,
                              srate: int,
                              overlap_factor: float,
                              segment_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """_summary_

    Args:
        X (np.ndarray): The epoched data in shape: nEpochs, nChans, nTime
        y (np.ndarray): The label array of shape nEpochs
        srate (int): the sampling rate
        segment_length (int, optional): The length of the new segment in seconds. Defaults to 1.
        overlap_factor (float, optional): The overlapping factor between segments. Defaults to 0.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The segmented X and y arrays
    """

    n_epochs, n_chans, n_time = X.shape

    assert segment_length <= n_time, "segment_length cannot be greater than the number of time points in epochs_data"
    assert X.ndim == 3, "X must be a 3-dimensional array"
    assert y.ndim == 1, "y must be a 1-dimensional array"
    assert len(X) == len(y), "X and y must have the same length"

    segment_length = int(segment_length * srate)  # get the index based on time lenght (e.g. 1s)
    overlap_length = int(segment_length * overlap_factor)
    stride = segment_length - overlap_length

    n_segments = (n_time - segment_length) // stride + 1  # number of segments per epoch
    total_segments = n_segments * n_epochs

    # Check if the remaining time points are enough for an additional segment
    remaining_time = n_time - segment_length - (n_segments - 1) * stride

    total_segments = n_segments * n_epochs + (remaining_time >= segment_length)

    segmented_X = np.zeros(shape=(total_segments, n_chans, segment_length))
    segmented_y = np.zeros(shape=(total_segments), dtype=float)

    for n_epoch in range(n_epochs):
        epoch = X[n_epoch]
        label = y[n_epoch]

        for n_segment in range(n_segments):
            start_idx = n_segment * stride
            end_idx = start_idx + segment_length

            # Adjust the segment end if it exceeds the available time points
            if end_idx > n_time:
                end_idx = n_time

            segmented_X[n_epoch * n_segments + n_segment, :, :] = epoch[:, start_idx:end_idx]
            segmented_y[n_epoch * n_segments + n_segment] = label

    segmented_y = segmented_y.astype(float)  # ensure that the label data type is of type float

    return segmented_X, segmented_y


def to_one_hot(y, by_sub=False):
    if by_sub:
        new_array = np.array(["nan" for nan in range(len(y))])
        for index, label in enumerate(y):
            new_array[index] = ''.join([i for i in label if not i.isdigit()])
    else:
        new_array = y.copy()
    total_labels = np.unique(new_array)
    mapping = {}
    for x in range(len(total_labels)):
        mapping[total_labels[x]] = x
    for x in range(len(new_array)):
        new_array[x] = mapping[new_array[x]]

    return tf.keras.utils.to_categorical(new_array)


## CNN MODEL ##


# 1. load MNE preprocessed epochs (nExamples*nChannels*nTime)
# data = mne.read_epochs(...)

# 2. Apply feature selection
# 2.1 apply the cluster-based mask to VI
# sig_channels: [0, 1, 3, 4, 10, 12 ...]
# sig time points (1.5s, 3s)

# data = data.crop(1.5, 3.0)
# data = data.get_data()[:, sig_channels, :] (nExamples*nSigChannels*nSigTime)

# 3. apply data augumentation techinque (cropped training)

# data, labels = generate_segmented_data(..some arguments...) -> nAugExamples*nSigChannels*nSigTime, nSigTime is the cropped time increased by a factor of N
# e.g. 300*45*128, where 128 is 0.5s (3 examples from 1, with same label)

# X_train, y_train, X_test, y_test = train_test_split(data, labels, test_size=0.10, random_seed=1)

# X_train, y_train, X_val, y_val = train_test_split(X_train, y_train, test_size=0.10, random_seed=1)

# .. reshape the model to reflect the dimension: (nExamples*nTime*nChannels)

# 4 Instantiate the CNN model

# cnn = CNN(arguments)
# cnn.build()

# cnn.fit(X_train, X_val...)

# paste the code from file we reviewed together (training example)


# tip: create dummy data with same dimension and feed it to the network for troubleshooting.
dummy_data = np.random.randn(300, 512, 25)  # 200, 300, 9
dummy_labels = np.random.randint(0, 2, size=(300))
dummy_labels = to_one_hot(dummy_labels, by_sub=False)

# Separate out the training set (80), validation set(10) and test set(10)
X_train, X_test, y_train, y_test = train_test_split(dummy_data, dummy_labels, test_size=0.10, shuffle=True,
                                                    random_state=1)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.10, shuffle=True, random_state=1)
input_shape = (None, 512, 25)

learning_rate = 1e-4

loss = tf.keras.losses.binary_crossentropy
optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
model = HopefullNet(inp_shape=input_shape)
model.build(input_shape)
modelPath = os.path.join(os.getcwd(), 'braincontrol_ml', 'notebooks', 'diane_time_frequency', "models", "test")

model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

checkpoint = ModelCheckpoint(  # set model saving checkpoints
    modelPath,  # set path to save model weights
    monitor='val_loss',  # set monitor metrics
    verbose=1,  # set training verbosity
    save_best_only=True,  # set if want to save only best weights
    save_weights_only=False,  # set if you want to save only model weights
    mode='auto',  # set if save min or max in metrics
    period=1  # interval between checkpoints
)

earlystopping = EarlyStopping(
    monitor='val_loss',  # set monitor metrics
    min_delta=0.001,  # set minimum metrics delta
    patience=4,  # number of epochs to stop training
    restore_best_weights=True,  # set if use best weights or last weights
)
callbacksList = [checkpoint, earlystopping]  # build callbacks list

hist = model.fit(X_train, y_train, epochs=1, batch_size=10,
                 validation_data=(X_val, y_val), callbacks=callbacksList)  # 32

loss = hist.history['loss']
acc = hist.history['accuracy']
val_loss = hist.history['val_loss']
val_acc = hist.history['val_accuracy']

print()
print(f'train_loss: {loss}')

# model.save_weights(checkpoint_path.format(epoch=0))

save_path = os.path.join(os.getcwd(), 'braincontrol_ml', 'notebooks', 'diane_time_frequency', "models", "test")
if not os.path.exists(save_path):
    os.mkdir(save_path)
with open(os.path.join(save_path, "hist.pkl"), "wb") as file:
    pickle.dump(hist.history, file)

# TESTING THE MODEL
del model  # Delete the original model, just to be sure!
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
