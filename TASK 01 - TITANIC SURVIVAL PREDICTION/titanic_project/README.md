# TASK 01-TITANIC SURVIVAL PREDICTION

Use the Titanic dataset to build a model that predicts whether a
passenger on the Titanic survived or not. This is a classic beginner
project with readily available data.
The dataset typically used for this project contains information
about individual passengers, such as their age, gender, ticket
class, fare, cabin, and whether or not they survived⛴️.

[CLICK HERE TO DOWNLOAD](https://www.kaggle.com/datasets/yasserh/titanic-dataset)

## Installation
⚠️ Before Moving Forward: Ensure Installation Necessary Libraries

🖥️ Open a Terminal or Command Prompt and Execute the Following Commands:

To install pandas:
```bash
pip install pandas
```
To install scikit-learn:
```bash
pip install scikit-learn
```
To install matplotlib:
```bash
pip install matplotlib
```
To install seaborn:
```bash
pip install seaborn
```
To install scikit-optimize:
```bash
pip install scikit-optimize
```
To install numpy:
```bash
pip install numpy
```

## CODE

```python
#CodSoft
#Task 01 - TITANIC SURVIVAL PREDICTION

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv('Titanic-Dataset.csv')

data = data.drop(columns=['Name', 'Ticket', 'Cabin'])
data['Age'] = data['Age'].fillna(data['Age'].median())
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

X = data.drop(columns=['Survived'])
y = data['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(max_depth=5)
model.fit(X_train, y_train)

train_accuracy = accuracy_score(y_train, model.predict(X_train))
test_accuracy = accuracy_score(y_test, model.predict(X_test))

print(f'Train Accuracy: {train_accuracy:.2f}')
print(f'Test Accuracy: {test_accuracy:.2f}')
```
🚀 To Start: Open a New Terminal in Visual Studio - Terminal -> New Terminal
```bash
python titanic.py
```
## Working
1. Importing Libraries:

pandas: Used for data manipulation and analysis.
train_test_split: From sklearn.model_selection, used to split the dataset into training and testing sets.
DecisionTreeClassifier: From sklearn.tree, used to create a decision tree classifier model.
accuracy_score: From sklearn.metrics, used to calculate the accuracy of the model.

2.Loading the Dataset:

Reads the Titanic dataset from a CSV file named 'Titanic-Dataset.csv' using pd.read_csv() function and stores it in the variable data.
Preprocessing the Data:

Drops unnecessary columns ('Name', 'Ticket', 'Cabin') from the dataset using drop(columns=[]).
Fills missing values in the 'Age' column with the median age using fillna().
Fills missing values in the 'Embarked' column with the most common port of embarkation using fillna() and mode().
Encodes categorical variables 'Sex' and 'Embarked' into numerical values using map().

3.Splitting the Dataset:

Separates the dataset into features (X) and the target variable (y) where X contains all columns except 'Survived' and y contains only the 'Survived' column.
Splits the dataset into training and testing sets using train_test_split() function with a test size of 20% and a random state of 42.

4.Model Training:

Initializes a decision tree classifier model with a maximum depth of 5 using DecisionTreeClassifier(max_depth=5).
Trains the model on the training data using fit() method.

5.Model Evaluation:

Calculates the accuracy of the trained model on both the training and testing datasets using accuracy_score() function.
The training accuracy is calculated by comparing the actual labels (y_train) with the predicted labels on the training data (model.predict(X_train)).
The testing accuracy is calculated similarly using the testing data.

6.Printing Results:

Prints the training and testing accuracies.
## Contributing

Pull requests are welcome! For significant alterations, kindly open an issue beforehand to discuss the proposed changes.

Please ensure the appropriate updates to tests. ✅

## License

[MIT](https://choosealicense.com/licenses/mit/)
