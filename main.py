import pandas as pd
import numpy as np
import random


def split_data(features, labels, train_ratio = 0.8):
    n_samples = len(features)
    indices = list(range(n_samples))
    random.shuffle(indices) # random shuffle of indices to split into test and training
    n_train = int(n_samples * train_ratio)
    train_indices = indices[:n_train] # first 120 instance of randomly shuffled index numbers
    test_indices = indices[n_train:] # last 30 instance of randomly shuffled index numbers
    train_features = features[train_indices]
    train_labels = labels[train_indices]  # division of test labels and train labels with features
    test_features = features[test_indices]
    test_labels = labels[test_indices]

    return train_features, train_labels, test_features, test_labels

def get_dis(x, y):
    return np.sqrt(np.sum((x - y) ** 2))

def knn(train_features, train_labels, test, k): 
    distances = []
    for i in range(len(train_features)):
        dist = get_dis(test, train_features[i]) # euclidean distance of train data vs test data
        distances.append((dist, train_labels[i])) # appends to dist array with train data label 
    distances.sort(key = lambda x: x[0]) # sorts in increasing order of first value 
    k_nearest = distances[:k] # take first k instances
    votes = {}
    for _, label in k_nearest:
        if label in votes: # if both labels match
            votes[label] += 1
        else:
            votes[label] = 1
    return max(votes.keys(), key = lambda x: votes[x]) # gives out maximum voted label
    
def get_conf_mat(true_labels, predicted_labels):
    classes = list(range(3))
    mat = np.zeros((3, 3), dtype = int) # filled conf matrix with 0s
    for true, pred in zip(true_labels, predicted_labels):
        mat[int(true)][int(pred)] += 1 # increases value if correct prediction found
    return mat

def calc_matrices(conf_mat): # calculates reports based on confusion matrix
    accuracy = np.trace(conf_mat) / np.sum(conf_mat) # diagonal sum / total sum
    sensitivity = []
    for i in range(3):
        row_sum = np.sum(conf_mat[i, :]) # row sum of confusion matriox
        if row_sum == 0:
            sensitivity.append(0.0)
        else:
            sensitivity.append(conf_mat[i, i] / row_sum)
    precision = []
    for i in range(3):
        col_sum = np.sum(conf_mat[:, i]) # column sum of confusion matrix
        if col_sum == 0:
            precision.append(0.0)
        else:
            precision.append(conf_mat[i, i] / col_sum)
    return accuracy, sensitivity, precision


def main():
    random.seed(50) # change random seed value to generate different data for tests and training
    np.random.seed(50)
    df = pd.read_excel("DataHW2.xlsx") # load excel to numpy
    #print(f"Loaded data shape: {df.shape}")
    #print(f"columns: {df.columns.tolist()}")

    features = df.iloc[1:, :4].values
    labels = df.iloc[1:, 4].values

    #print(f"length of dataset loaded: {len(features)}")
    #print(f"features shape: {features.shape}")

    train_features, train_labels, test_features, test_labels = split_data(features, labels, 0.8) # change value of 0.8 to split dataset into different distributions
    k = 10  # change value of k for KNN
    predictions = []
    for test in test_features:
        pred = knn(train_features, train_labels, test, k) # prediction for current single test data
        predictions.append(pred)
    predictions = np.array(predictions) # all class predictions in test data
    conf_mat = get_conf_mat(test_labels, predictions) 
    #  print(conf_mat)
    accuracy, sensitivity, precision = calc_matrices(conf_mat)
    print(f"k = {k}")
    print("confusion matrix is: ")
    for row in conf_mat:
        print(' '.join(map(str, row)))
    print(f"accuracy = {accuracy:.3f}")
    class_names = ['Setosa', 'Versicolor', 'Virginica']
    for i, class_name in enumerate(class_names):
        print(f"Sensitivity_{class_name} = {sensitivity[i]:.3f}")
    for i, class_name in enumerate(class_names):
        print(f"precision_{class_name}: {precision[i]:.3f}")


if __name__ == "__main__":
    main()