<kbd> Lux-Academy-Project-1</kbd>
Telecom Customer Churn Prediction
Overview
This Python script is designed to predict customer churn for a telecom company using machine learning. It utilizes a dataset containing customer information, including various features such as account length, area code, international plan, voice mail plan, and more. The goal is to build a predictive model that can identify customers who are likely to leave (churn) and take proactive measures to retain them.

Prerequisites
Before running the code, ensure you have the following libraries installed in your Python environment:

pandas
numpy
matplotlib
scikit-learn
You can install these libraries using pip if they are not already installed:

bash
Copy code
pip install pandas numpy matplotlib scikit-learn
Usage
Data Preparation:

Store your dataset in a folder named "Data" within the project directory.
Update the data_folder and data_file variables to specify the path to your data file.
Running the Code:

Open the Python script (e.g., churn_project.py) in your preferred code editor (e.g., Visual Studio Code).
Run the script to execute the following steps:
Data Loading and Preprocessing:

Load the dataset from the specified path.
Check for missing values and encode categorical variables using label encoding.
Drop non-numeric columns, such as "State," if they are not relevant for prediction.
Data Splitting and Scaling:

Split the data into training and testing sets for model evaluation.
Scale the numeric features to ensure they are on the same scale using StandardScaler.
Model Training and Evaluation:

Train a Random Forest Classifier on the training data.
Evaluate the model's performance on the testing data, providing accuracy, ROC AUC, classification report, and confusion matrix.
Feature Importance:

Visualize feature importance to understand which features are most influential in predicting customer churn.
Results
The script provides insights into customer churn prediction, allowing the telecom company to take proactive measures to retain valuable customers. It also identifies the most important factors contributing to churn.

Contributors
[Your Name]
License
This project is licensed under the [License Name] - see the LICENSE.md file for details.

