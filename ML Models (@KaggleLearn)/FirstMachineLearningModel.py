import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

# TODO: Can use input lines (lines waiting for input) to pause within the program, maybe?

# 1. Get dataset.
# insert filepath here
print("Where is your dataset? Please copy absolute file path in the input the follows: ")
# TODO: Put in way of inputting absolute path for a dataset.

dataset_file_path = ''
dataset_data = pd.read_csv()


# 2. Dropping missing values with dropna
dataset_data = dataset_data.dropna(axis=0) 

# 3. Setting target.

print(dataset_set.columns)
print("Which column is your target? ")
# TODO: Put in way of inputting column(s) for target.
# y = 

# 4. Choosing features

dataset_features = ['', '', '']
X = dataset_data[dataset_features]

# 5. Viewing Data

print("The head of the features of the dataset: (These are the first five rows of parameters that the model will be using to predict target values.) \n")
X.head()


# TODO: Check if string values show up in the description, or if it gets processed differently. 
print("The head of the features of the dataset: (This is the statistical analysis of those features.) \n")
X.describe()

# 6. Predict using Random Forest model:

# random_state ensures you get the same results in each run by allowing the same degree of randomness, keep this value consistent, or just keep it at 1
dataset_model = DecisionTreeRegressor(random_state=1)
dataset_model.fit(X, y)

print("Making predictions for the following 5 values:")
print(X.head())
print("The predictions are: ")
print(dataset_data.predict(X.head()))


