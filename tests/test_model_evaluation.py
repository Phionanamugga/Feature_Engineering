import pytest
from sklearn.metrics import accuracy_score

# Use the data fixture to get the test data and the model fixture to get the model
def test_model_accuracy(data, model):
    x_train, x_test, y_train, y_test = data
    # Train the model with x_train and y_train
    model.fit(x_train, y_train)
    
    # Make predictions on x_test
    y_pred = model.predict(x_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Assert that the accuracy is above a threshold
    assert accuracy > 0.78, f"Model accuracy too low: {accuracy}"
