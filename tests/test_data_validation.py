import pytest

# Use the data fixture to get the training and testing data
def test_data_not_empty(data):
    x_train, x_test, y_train, y_test = data
    # Ensure x_train and y_train are not empty
    assert x_train is not None and len(x_train) > 0
    assert y_train is not None and len(y_train) > 0

def test_no_missing_values(data):
    x_train, x_test, y_train, y_test = data
    # Check that x_train does not have any missing values
    assert x_train.isnull().sum().sum() == 0
