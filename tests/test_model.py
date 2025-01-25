import pytest

def test_model_training(data, model):
    x_train, x_test, y_train, y_test = data

    # Train the model
    model.fit(x_train, y_train)

    # Check if the model can make predictions
    predictions = model.predict(x_test)

    # Ensure the predictions have the same length as the test set
    assert len(predictions) == len(y_test)


def test_model_accuracy(data, model):
    x_train, x_test, y_train, y_test = data

    # Train the model
    model.fit(x_train, y_train)

    # Calculate accuracy
    accuracy = model.score(x_test, y_test)

    # Ensure accuracy is a reasonable number (e.g., between 0 and 1)
    assert 0 <= accuracy <= 1
