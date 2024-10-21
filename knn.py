import numpy as np
import pandas as pd



# Preprocessing and KNN implementation
def preprocess_data(data):

    data = data.drop(columns=['Surname', 'CustomerId'])
    data = pd.get_dummies(data, columns=['Geography', 'Gender'], drop_first=True)
    data.fillna(data.median(), inplace=True)
    
    return data

def min_max_scaler(data):
    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)
    scaled_data = (data - min_val) / (max_val - min_val)
    return scaled_data

# KNN from scratch
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b)**2))

def knn_predict(X_train, y_train, X_test, k=30):
    print(f"Starting KNN predictions with k={k}...")
    predictions = []
    for test_point in enumerate(X_test):
        distances = []
        for i, train_point in enumerate(X_train):
            distance = euclidean_distance(train_point, test_point)
            distances.append((distance, y_train[i]))
        
        # Sort by distance and get the k nearest neighbors
        distances.sort(key=lambda x: x[0])
        k_nearest_neighbors = distances[:k]
        
        # Majority vote for classification (average of class labels to get probability)
        class_votes = [neighbor[1] for neighbor in k_nearest_neighbors]
        prediction = np.mean(class_votes)  # Probability for churn
        predictions.append(prediction)
    
    return np.array(predictions)

# Load train and test datasets
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# 1. Preprocess train and test data
train_data = preprocess_data(train_data)
test_data = preprocess_data(test_data)

# 2. Scale numerical features
numerical_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
train_data[numerical_features] = min_max_scaler(train_data[numerical_features].values)
test_data[numerical_features] = min_max_scaler(test_data[numerical_features].values)

# 3. Prepare training and test datasets
X_train = train_data.drop(columns=['Exited', 'id']).values
y_train = train_data['Exited'].values
X_test = test_data.drop(columns=['id']).values

# 4. Train and predict with KNN
k = 5  # You can adjust k to optimize performance
test_predictions = knn_predict(X_train, y_train, X_test, k=k)

# 5. Prepare submission file (only 'id' and 'Exited')
submission = pd.DataFrame({'id': test_data['id'], 'Exited': test_predictions})

# Save the submission file
submission.to_csv('submission.csv', index=False)