# Import the libraries
import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn import model_selection
import os
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.regularizers import l2


BASE_DIR = os.getcwd()

# Import the training dataset
dataset = pd.read_csv('./training_set.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


# apply PCA to reduce the dimension
def principal_component_analysis(feature_set):
    std_scaler = preprocessing.StandardScaler()
    std_feature_matrix = std_scaler.fit_transform(feature_set)
    std_feature_matrix = pd.DataFrame(std_feature_matrix)

    pca = PCA(n_components=39)
    principal_components = pca.fit_transform(std_feature_matrix)
    return principal_components


reduced_data = principal_component_analysis(X)
# shape : (3910, 39)


# Reserve 1000 samples for validation
X_test = reduced_data[-728:]
y_test = y[-728:]
x_train = reduced_data[:-728]
y_train = y[:-728]

"""
X_test: (728, 39)
y_test: (728,)
x_train:(3182, 39)
y_train:(3182,)
"""

# splitting the dataset
X_train, X_validate, Y_train, Y_validate = model_selection.train_test_split(x_train, y_train, test_size=0.2)

"""
shape:
X_train:    (2545, 39)
X_validate: (637, 39)
Y_train:    (2545,)
Y_validate: (637,)
"""

# One-hot encoding the output
num_classes = 2
y_train_encoded = keras.utils.to_categorical(Y_train, num_classes)
y_validate_encoded = keras.utils.to_categorical(Y_validate, num_classes)
y_test_encoded = keras.utils.to_categorical(y_test, num_classes)

# build the model
model = Sequential()
model.add(Dense(528, activation='tanh', input_dim=X_train.shape[1], kernel_regularizer=l2(0.001)))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

"""
model.summary():

Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense (Dense)                (None, 528)               21120
_________________________________________________________________
dropout (Dropout)            (None, 528)               0
_________________________________________________________________
dense_1 (Dense)              (None, 2)                 1058
=================================================================
Total params: 22,178
Trainable params: 22,178
Non-trainable params: 0
_________________________________________________________________

"""

# Compiling the model using categorical_crossentropy loss, and rmsprop optimizer.
opt = keras.optimizers.RMSprop(learning_rate=1e-4)
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])


# train the model
history = model.fit(
    X_train,
    y_train_encoded,
    batch_size=16,
    epochs=150,
    # We pass some validation for
    # monitoring validation loss and metrics
    # at the end of each epoch
    validation_data=(X_validate, y_validate_encoded),
)

"""
Epoch 150/150
accuracy: 0.9289 - val_loss: 0.2276 - val_accuracy: 0.9278

"""

# Evaluate the model on the test data using `evaluate`
results = model.evaluate(X_test, y_test_encoded)
"""
[test loss: 0.2684154212474823, test accuracy: 0.9093406796455383]


"""

model.save(BASE_DIR+'/arya_model.h5')