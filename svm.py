import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_svm(features, labels):
    """
    Trains an SVM model.

    Args:
        features (numpy.ndarray): Feature vectors.
        labels (numpy.ndarray): Corresponding labels.

    Returns:
        sklearn.svm.SVC: Trained SVM model.
    """
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Create an SVM classifier
    clf = svm.SVC(kernel='linear', C=1)

    # Train the classifier
    clf.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = clf.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

    return clf

if __name__ == '__main__':
    # Generate some dummy data for demonstration
    X = np.random.rand(100, 10)  # 100 samples, 10 features
    y = np.random.randint(0, 2, 100)  # 100 labels, 0 or 1

    # Train the SVM model
    model = train_svm(X, y)

    print("SVM model trained successfully.")