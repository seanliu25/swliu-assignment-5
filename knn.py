import numpy as np
import pandas as pd

# Preprocessing and KNN implementation
def preprocess_data(data):
    data = data.drop(columns=['Surname', 'CustomerId'])

    data = pd.get_dummies(data, columns=['Geography', 'Gender'], drop_first=True)

    # Fill missing values with the median (if any)
    data.fillna(data.median(), inplace=True)

    return data

def min_max_scaler(data):
    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)
    scaled_data = (data - min_val) / (max_val - min_val)
    return scaled_data

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b)**2))

def knn_predict(X_train, y_train, X_test, k=5, weighted=False):
    print(f"Starting KNN predictions with k={k}...")
    predictions = []
    for idx, test_point in enumerate(X_test):
        if idx % 50 == 0:
            print(f"Predicting test point {idx + 1}/{len(X_test)}...")

        distances = []
        for i, train_point in enumerate(X_train):
            distance = euclidean_distance(train_point, test_point)
            distances.append((distance, y_train[i]))

        distances.sort(key=lambda x: x[0])
        k_nearest_neighbors = distances[:k]

        if weighted:
            weights = [1 / (dist[0] + 1e-5) for dist in k_nearest_neighbors]
            weighted_sum = sum(weight * label for weight, (_, label) in zip(weights, k_nearest_neighbors))
            prediction = weighted_sum / sum(weights)
        else:
            class_votes = [neighbor[1] for neighbor in k_nearest_neighbors]
            prediction = np.mean(class_votes)  
        
        predictions.append(prediction)

    print("KNN predictions complete.")
    return np.array(predictions)

print("Loading datasets...")
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
print("Datasets loaded.")

train_data = preprocess_data(train_data)
test_data = preprocess_data(test_data)

numerical_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
train_data[numerical_features] = min_max_scaler(train_data[numerical_features].values)
test_data[numerical_features] = min_max_scaler(test_data[numerical_features].values)

X_train = train_data.drop(columns=['Exited', 'id']).values
y_train = train_data['Exited'].values
X_test = test_data.drop(columns=['id']).values


k = 15
weighted = True  # Use weighted distance voting
test_predictions = knn_predict(X_train, y_train, X_test, k=k, weighted=weighted)

submission = pd.DataFrame({'id': test_data['id'], 'Exited': test_predictions})

submission.to_csv('submission.csv', index=False)