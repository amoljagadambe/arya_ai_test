# Import the libraries
import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA
from keras import models
import os

BASE_DIR = os.getcwd()

# Import the test dataset
X_predict = pd.read_csv('./test_set.csv')


# apply PCA to reduce the dimension
def principal_component_analysis(feature_set):
    std_scaler = preprocessing.StandardScaler()
    std_feature_matrix = std_scaler.fit_transform(feature_set)
    std_feature_matrix = pd.DataFrame(std_feature_matrix)

    pca = PCA(n_components=39)
    principal_components = pca.fit_transform(std_feature_matrix)
    return principal_components


# It can be used to reconstruct the model identically.
reconstructed_model = models.load_model(BASE_DIR + '/arya_model.h5')

# reduced the data
X_test = principal_component_analysis(X_predict)

prediction = reconstructed_model.predict(X_test)

output_label = []
for pred in prediction:
    y_classes = pred.argmax()
    output_label.append(y_classes)
    # df = pd.DataFrame([y_classes], columns=['predictions'])
    # print(df)
    # df.to_csv(BASE_DIR + '/prediction_results.csv')

df = pd.DataFrame(output_label, columns=['Label'])

df.to_csv(BASE_DIR + '/predictions.csv')