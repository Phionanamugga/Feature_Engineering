# Machine Learning Pipeline and Model Training 

This app demonstrates a machine learning pipeline for preprocessing, training, and evaluating a classification model. It uses the Support Vector Classifier (SVC) from sklearn to classify data with categorical and numerical features.

## Features
### Data Preprocessing:
Identification of categorical and numerical features from the dataset.
Scaling numerical features using StandardScaler.
Encoding categorical features using OneHotEncoder with handle_unknown="ignore".

### Pipeline Implementation:
Combined preprocessing steps using ColumnTransformer.
Ensures consistent transformations for both training and testing datasets.

### Model Training:
A Support Vector Classifier (SVC) is trained using preprocessed data.

### Model Evaluation:
Evaluates model accuracy on a test dataset.
Reports overall accuracy.

## Installation

Prerequisites
Python >= 3.8
Required Python libraries:
scikit-learn
numpy
pandas (if used for data handling)
Install the dependencies via pip:

pip install scikit-learn pandas numpy

## Usage
1. Data Preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

### Identify categorical and numerical columns
categorical_attributes = x_train.select_dtypes(include=["object"]).columns
numerical_attributes = x_train.select_dtypes(include=["int64", "float64"]).columns

### Create a ColumnTransformer
ct = ColumnTransformer([
    ('scaling', StandardScaler(), numerical_attributes),
    ('encoding', OneHotEncoder(handle_unknown="ignore"), categorical_attributes)
])

### Fit the transformer and preprocess the data
ct.fit(x_train)
x_train = ct.transform(x_train)
x_test = ct.transform(x_test)
2. Model Training
import sklearn.svm

model = sklearn.svm.SVC()
model.fit(x_train, y_train)
3. Model Evaluation
import sklearn.metrics

y_predicted = model.predict(x_test)
accuracy = sklearn.metrics.accuracy_score(y_test, y_predicted)
print("Model Accuracy:", accuracy)
Example Output

After preprocessing, the transformed datasets have the following shapes:
x_train size: (22643, 104)
x_test size: (7519, 104)
Model accuracy after training:
Accuracy: 85.42%

## Notes

ColumnTransformer: This tool applies separate transformations to different columns (e.g., scaling numerical values and encoding categorical features).
Support Vector Classifier: The default kernel is rbf, which works well for non-linear classification problems.
Accuracy: This metric evaluates the percentage of correctly classified instances. Consider using additional metrics like precision, recall, and F1-score for a more comprehensive evaluation.

## Future Improvements
### Data Validation:
Implement checks for missing values or invalid data types in input datasets.

### Hyperparameter Tuning:
Improve model performance using techniques like Grid Search or Randomized Search for hyperparameter optimization.

### Pipeline Integration:
Use Pipeline from sklearn.pipeline to streamline preprocessing and model training steps.

### Additional Metrics:
Evaluate the model using confusion matrices, precision, recall, and F1-score.

## License
This app is provided under the MIT License. Feel free to use, modify, and distribute.