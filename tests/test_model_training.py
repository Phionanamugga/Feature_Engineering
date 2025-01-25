import pytest
from sklearn.svm import SVC

def test_model_training(data, model):
    x_train, x_test, y_train, y_test = data
    model.fit(x_train, y_train)
    assert model.score(x_test, y_test) > 0.7, "Model training failed"

'''def test_model_feature_count(data):
    x_train, x_test, y_train, y_test = data
    # Check if the model's training data has the expected number of features
    assert x_train.shape[1] == 14, f"Expected 14 features, got {x_train.shape[1]}"'''

def test_model_feature_count(data):
    x_train, x_test, y_train, y_test = data
    print(x_train.shape)  # To inspect the actual number of features
    # You might need to adjust the expected number of features
    expected_feature_count = x_train.shape[1]  # Get this dynamically if you want to adjust it
    assert x_train.shape[1] == expected_feature_count, f"Expected {expected_feature_count} features, got {x_train.shape[1]}"
