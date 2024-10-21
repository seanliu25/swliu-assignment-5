import numpy as np
import pandas as pd

def train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    indices = np.random.permutation(len(X))
    test_set_size = int(len(X) * test_size)
    test_indices = indices[:test_set_size]
    train_indices = indices[test_set_size:]
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    return X_train, X_test, y_train, y_test

class Node:
    def __init__(self, gini, num_samples, num_samples_per_class, predicted_class):
        self.gini = gini
        self.num_samples = num_samples
        self.num_samples_per_class = num_samples_per_class
        self.predicted_class = predicted_class
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None

class DecisionTreeClassifierManual:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y):
        y = y.astype(int)
        self.n_classes = len(set(y))
        self.tree = self._grow_tree(X, y)

    def predict_proba(self, X):
        return [self._predict_proba(inputs) for inputs in X]

    def predict(self, X):
        return [1 if prob >= 0.5 else 0 for prob in self.predict_proba(X)]

    def _predict_proba(self, inputs):
        node = self.tree
        while node.left:
            if inputs[node.feature_index] < node.threshold:
                node = node.left
            else:
                node = node.right
        prob = node.num_samples_per_class[1] / sum(node.num_samples_per_class)
        return prob

    def _grow_tree(self, X, y, depth=0):
        num_samples_per_class = [np.sum(y == i) for i in range(self.n_classes)]
        predicted_class = np.argmax(num_samples_per_class)
        node = Node(
            gini=self._gini(y),
            num_samples=len(y),
            num_samples_per_class=num_samples_per_class,
            predicted_class=predicted_class,
        )

        # If stopping criteria is met, return the node
        if depth >= self.max_depth or len(y) < self.min_samples_split or np.unique(y).size == 1:
            return node

        idx, thr = self._best_split(X, y)
        if idx is None:
            return node

        # Recursively split left and right
        indices_left = X[:, idx] < thr
        X_left, y_left = X[indices_left], y[indices_left]
        X_right, y_right = X[~indices_left], y[~indices_left]
        node.feature_index = idx
        node.threshold = thr
        node.left = self._grow_tree(X_left, y_left, depth + 1)
        node.right = self._grow_tree(X_right, y_right, depth + 1)
        return node

    def _gini(self, y):
        m = len(y)
        return 1.0 - sum((np.sum(y == c) / m) ** 2 for c in range(self.n_classes))

    def _best_split(self, X, y):
        m, n = X.shape
        if m <= 1:
            return None, None

        num_parent = [np.sum(y == c) for c in range(self.n_classes)]
        best_gini = 1.0 - sum((num / m) ** 2 for num in num_parent)
        best_idx, best_thr = None, None

        for idx in range(n):
            thresholds, classes = zip(*sorted(zip(X[:, idx], y)))
            num_left = [0] * self.n_classes
            num_right = num_parent.copy()

            for i in range(1, m):
                c = int(classes[i - 1])
                num_left[c] += 1
                num_right[c] -= 1
                gini_left = 1.0 - sum((num_left[x] / i) ** 2 for x in range(self.n_classes))
                gini_right = 1.0 - sum((num_right[x] / (m - i)) ** 2 for x in range(self.n_classes))
                gini = (i * gini_left + (m - i) * gini_right) / m

                if thresholds[i] == thresholds[i - 1]:
                    continue

                if gini < best_gini:
                    best_gini = gini
                    best_idx = idx
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2

        return best_idx, best_thr

def accuracy_score_manual(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
   
    # Print shapes and data types for debugging
    print("y_true shape:", y_true.shape, "y_pred shape:", y_pred.shape)
    print("y_true dtype:", y_true.dtype, "y_pred dtype:", y_pred.dtype)
   
    if y_true.shape != y_pred.shape:
        raise ValueError("Shapes of y_true and y_pred must match!")
   
    # Perform elementwise comparison and count the correct predictions
    correct = np.sum(y_true == y_pred)
    return correct / len(y_true)



def preprocess_data(data):
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    data.fillna(data[numeric_columns].median(), inplace=True)
    data = pd.get_dummies(data, columns=['Geography', 'Gender'], drop_first=True)
    return data

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Preprocess data
train_data = preprocess_data(train_data)
test_data = preprocess_data(test_data)

X = train_data.drop(columns=['Exited', 'id', 'CustomerId', 'Surname']).values
y = train_data['Exited'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifierManual(max_depth=7, min_samples_split=20)
clf.fit(X_train, y_train)

y_train_pred = clf.predict(X_train)
train_accuracy = accuracy_score_manual(y_train, y_train_pred)

y_test_pred = clf.predict(X_test)
print("y_test:", y_test)
print("y_test_pred:", y_test_pred)
test_accuracy = accuracy_score_manual(y_test, y_test_pred)


print(f"Train Accuracy: {train_accuracy:.3f}")
print(f"Test Accuracy: {test_accuracy:.3f}")

X_final_test = test_data.drop(columns=['id', 'CustomerId', 'Surname']).values
test_probabilities = clf.predict_proba(X_final_test)

submission = pd.DataFrame({'id': test_data['id'], 'Exited': test_probabilities})
submission.to_csv('submission.csv', index=False)

print("Submission2 file created.")