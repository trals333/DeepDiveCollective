import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import datetime
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
import pickle

df_cab = pd.read_csv('cab_rides.csv')
df_cab.reset_index(drop=True, inplace=True)

print("Taxi fare dataset\n")
print(df_cab.info())

print("Shape ", df_cab.shape)

df_cab = df_cab.iloc[:, :]
print("\n", df_cab.head(2))
print("\n", df_cab.describe())

df_weather = pd.read_csv('weather.csv')

print("Weather dataset\n")
print(df_weather.info())

print("Shape ", df_weather.shape)

df_weather = df_weather.iloc[:, :]
print("\n", df_weather.head(2))
print("\n", df_weather.describe())


print("Checking null values: \n", df_cab.isnull().sum())

# convert 13digit time stamp to datetime format
df_cab['date_time']= pd.to_datetime(df_cab['time_stamp']/1000, unit='s')
df_cab['date']= df_cab['date_time'].dt.date
df_cab['day'] = df_cab.date_time.dt.dayofweek
df_cab['hour'] = df_cab.date_time.dt.hour

df_cab['fare_per_mile']= round(df_cab.price/df_cab.distance,2)

del df_cab['time_stamp'] # deleting unwanted time_stamp

df_cab['fare_per_mile'] = df_cab['fare_per_mile'].astype(float)
df_cab['fare_per_mile'].fillna('2.8',inplace=True)
# df_cab['price'] = df_cab['fare_per_mile'] * df_cab['distance']

df_cab['fare_per_mile'] = df_cab['fare_per_mile'].astype(float)
print(df_cab['fare_per_mile'].dtype)

df_cab['price'] = df_cab['fare_per_mile'] * df_cab['distance']

print("Checking null values: \n", df_weather.isnull().sum())

df_weather['rain'].fillna(0, inplace = True)
df_weather['date_time'] = pd.to_datetime(df_weather['time_stamp'], unit='s')
del df_weather['time_stamp']

df_cab['merge_date'] = df_cab.source.astype(str) +" - "+ df_cab.date_time.dt.date.astype("str") +" - "+ df_cab.date_time.dt.hour.astype("str")
df_weather['merge_date'] = df_weather.location.astype(str) +" - "+ df_weather.date_time.dt.date.astype("str") +" - "+ df_weather.date_time.dt.hour.astype("str")

df_weather = df_weather.groupby(['merge_date'])
columns_to_mean = ['temp', 'clouds', 'pressure', 'rain', 'humidity', 'wind']
df_weather = df_weather[columns_to_mean].mean()
df_weather.reset_index(inplace=True)

df_merged = pd.merge(df_cab, df_weather, on='merge_date')
print("Shape after merging both datasets", df_merged.shape)

# Removing unwanted columns
df_merged = df_merged.drop(['date_time','id','product_id','fare_per_mile','surge_multiplier','destination','source','date'], axis=1)
df_merged = df_merged.loc[:, df_merged.columns !='merge_date']

# final_dataset = df_merged.drop(['cab_type'],axis=1)
final_dataset = df_merged
final_dataset.info()


# Load the dataset
data = final_dataset

# Select weather-related features and the target variable
# features = ['temp', 'clouds', 'pressure', 'rain', 'humidity', 'wind']
features = ['distance']
target = 'price'

# Ensure no missing values
data = data.dropna(subset=features + [target])

# Split data into features (X) and target (y)
X = data[features]
y = data[target]

# Split the data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the regressor with hyperparameters
regressor = DecisionTreeRegressor(random_state=42, min_samples_leaf=10, min_samples_split=2, max_depth=10)

# Train the model
regressor.fit(X_train, y_train)

from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import make_scorer, mean_squared_error

# Initialize the model
dt = DecisionTreeRegressor(random_state=42)

# Define the hyperparameter grid
param_grid = {
    'max_depth': [3, 5, 10, 20, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 10],
    'max_features': [None, 'sqrt', 'log2']
}

# Use a scoring metric (negative MSE here)
scorer = make_scorer(mean_squared_error, greater_is_better=False)

# Perform Grid Search
grid_search = GridSearchCV(
    estimator=dt,
    param_grid=param_grid,
    scoring=scorer,
    cv=5,  # 5-fold cross-validation
    verbose=1,  # Show progress
    n_jobs=-1   # Use all available CPU cores
)

# Fit the Grid Search on training data
grid_search.fit(X_train, y_train)

# Output the best parameters and their performance
print("Best Hyperparameters:", grid_search.best_params_)
print("Best Cross-Validated MSE:", -grid_search.best_score_)


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Make predictions
y_pred = regressor.predict(X_test)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R² Score: {r2}")

plt.scatter(y_test, y_pred, color='blue', label='Testing data')

# Plotting the regression line
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='-', linewidth=2, label='Regression line')

# Adding labels and title
plt.title('Actual vs. Predicted')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.legend()

# Display the plot
plt.show()

LinearRegressionModel = LinearRegression()

LinearRegressionModel.fit(X_train, y_train)

y_pred = LinearRegressionModel.predict(X_test)

print('Mean Absolute Error: ', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error: ', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error: ', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
MAPE = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
print('Mean Absolute Percentage Error: ', MAPE)
print('\nFinal Accuracy: ', 100 - MAPE)

plt.scatter(y_test, y_pred, color='blue', label='Testing data')

# Plotting the regression line
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='-', linewidth=2, label='Regression line')

# Adding labels and title
plt.title('Actual vs. Predicted')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.legend()

# Display the plot
plt.show()

from sklearn.linear_model import Lasso

lasso = Lasso(alpha=11)  # alpha controls the penalty strength
lasso.fit(X_train, y_train)
y_pred_lasso = lasso.predict(X_test)

# Evaluate Lasso
print(f"Lasso R2 Score: {r2_score(y_test, y_pred_lasso):.2f}")

# Predicted prices using Lasso
y_pred_lasso = lasso.predict(X_test)

# Scatter plot of actual vs predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_lasso, alpha=0.7, color='purple')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2)  # y=x line
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices (Lasso Regression)")
plt.show()


dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Define parameters for XGBoost regression
params = {
    'objective': 'reg:squarederror',  # specify regression objective
    'eval_metric': 'rmse'  # use RMSE as evaluation metric
}

num_boost_round = 1000  # number of boosting rounds
XgboostModel = xgb.train(params, dtrain, num_boost_round=num_boost_round)

y_pred = XgboostModel.predict(dtest)

print('Mean Absolute Error: ', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error: ', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error: ', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
MAPE = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
print('Mean Absolute Percentage Error: ', MAPE)
print('\nFinal Accuracy: ', 100 - MAPE)


plt.scatter(y_test, y_pred, color='blue', label='Testing data')

# Plotting the regression line
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='-', linewidth=2, label='Regression line')

# Adding labels and title
plt.title('Actual vs. Predicted')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.legend()

# Display the plot
plt.show()


num_boost_round = 100
XgboostModel = xgb.train(params, dtrain, num_boost_round = num_boost_round)
y_pred = XgboostModel.predict(dtest)

print('Mean Absolute Error: ', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error: ', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error: ', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
MAPE = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
print('Mean Absolute Percentage Error: ', MAPE)
print('\nFinal Accuracy: ', 100 - MAPE)


plt.scatter(y_test, y_pred, color='blue', label='Testing data')

# Plotting the regression line
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='-', linewidth=2, label='Regression line')

# Adding labels and title
plt.title('Actual vs. Predicted')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.legend()

# Display the plot
plt.show()
