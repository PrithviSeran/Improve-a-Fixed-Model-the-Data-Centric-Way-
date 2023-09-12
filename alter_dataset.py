from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
#from IPython.display import display
import warnings
warnings.simplefilter("ignore")
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer


#creating function for loss calculating
def rmse(y_preds, y_true):
    return mean_squared_error(y_preds, y_true, squared = False)
 
#loading dataset
dataset = pd.read_csv('sample_submission.csv')

#loading another dataset to write to the csv file
dataset_copy = pd.read_csv('sample_submission.csv')

# creating the evaluating model to test the dataset
rf = RandomForestRegressor(
    n_estimators = 1000,
    max_depth = 7,
    n_jobs = -1,
    random_state = 42
)

# getting the train dataset 
X = dataset.drop('id', axis = 1)

# getting the y_truth variables
y = dataset.pop('target')

# getting the headers of the dataset (for graphing)
headers = dataset.columns.values

#training the random forest model
rf.fit(X, y)

#getting the feature importances manually 
custom_importance = permutation_importance(rf, X, y, scoring = make_scorer(rmse, greater_is_better = False), n_repeats=20, random_state=42)

#using the the built in function to get the feature importances
feature_importance = pd.DataFrame(rf.feature_importances_, index = list(X))

#plotting all the feature importances of the dataset
plt.figure(figsize = (15, 10), dpi = 75)
plt.bar(headers, feature_importance, color ='maroon',
        width = 0.4)
plt.show()

# From the graph, we can tell the the most important features of the datset are, O2_2, O2_1, BOD5_5
# So thats all the data we need, the other features are useless in determining the target variables
# We should also clip the target variables to get rid of all the anomalies and outliers in the dataset
# lets where most of the data in the target vector is

x_axis = np.array(range(3500))

y_axis = np.array(y)

plt.plot(x_axis, y_axis)
plt.show()

# most of the values for the targets are between 6 and 16, so lets clips all values beyond this intersection
dataset_copy["target"] = dataset_copy.target.clip(6, 16)

#turn every feature that is not important into 0
dataset_copy[list(dataset_copy.drop(['O2_1', 'O2_2', 'BOD5_5', 'id'], axis = 1))] = 0

#writing the dataset to a csv file
dataset_copy.to_csv("out.csv", index=False)



