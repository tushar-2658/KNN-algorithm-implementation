import pandas as pd
import numpy as np
import random


def split_data(features, labels, train_ratio = 0.8):
    n_samples = len(features)
    indices = list(range(n_samples))
    random.shuffle(indices)
    n_train = int(n_samples * train_ratio)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    train_features = features[train_indices]
    train_labels = labels[train_indices]
    test_features = features[test_indices]
    test_labels = labels[test_indices]

    return train_features, train_labels, test_features, test_labels

def get_dis(x, y):
    return np.sqrt(np.sum((x - y) ** 2))

def knn(train_features, train_labels, test, k): 
    distances = []
    for i in range(len(train_features)):
        dist = get_dis(test, train_features[i])
        distances.append((dist, train_labels[i]))
    distances.sort(key = lambda x: x[0])
    k_nearest = distances[:k]
    votes = {}
    for _, label in k_nearest:
        if label in votes:
            votes[label] += 1
        else:
            votes[label] = 1
    return max(votes.keys(), key = lambda x: votes[x])
    

def main():
    random.seed(42)
    np.random.seed(42)
    df = pd.read_excel("DataHW2.xlsx")
    print(f"Loaded data shape: {df.shape}")
    #print(f"columns: {df.columns.tolist()}")

    features = df.iloc[1:, :4].values
    labels = df.iloc[1:, 4].values

    print(f"length of dataset loaded: {len(features)}")
    print(f"features shape: {features.shape}")

    train_features, train_labels, test_features, test_labels = split_data(features, labels, 0.8)
    k = 10
    predictions = []
    for test in test_features:
        pred = knn(train_features, train_labels, test, k)
        predictions.append(pred)
    print(predictions)
    

if __name__ == "__main__":
    main()