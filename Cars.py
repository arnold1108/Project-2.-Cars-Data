from ipaddress import collapse_addresses
import pandas as pd
import numpy as np

cars = pd.read_csv('E:\Python\PROJECTS/Cars.csv')
print(cars)

#DATA EXPLORATION
print(cars.shape)
print(cars.describe())
print(cars.info())

# Data cearning
print(cars.isna())

# 1.  Find all Null Values in the dataset. If there is any null value in any column, then fill it with the mean of that column.
import seaborn as sns
import matplotlib.pyplot as plt
cars = cars.dropna(how='all') #Dropped the entirely empty rows. Now remaining with the ones with genuiely na values
sns.heatmap(cars.isna())
plt.show()
print(cars.columns[cars.isna().any()]) #Identifying the column with the missing values. We find the na values are in the Cylinders column

def Mean():
    mean = cars['Cylinders'].mean()
    return mean
cars = cars.fillna(Mean)
sns.heatmap(cars.isna())
plt.show()

# Q. 2) Question ( Based on Value Counts )- Check what are the different types of Make are there in our dataset. And, what is the count (occurrence) of each Make in the data ?
print(cars['Make'].unique())
print(cars['Make'].value_counts())

# Q. 3) Instruction ( Filtering ) - Show all the records where Origin is Asia or Europe.
origin = cars[(cars['Origin'] == 'Asia') | (cars['Origin'] == 'Europe')]
print(origin)

# Q. 4) Instruction ( Removing unwanted records ) - Remove all the records (rows) where Weight is above 4000.
cars = cars[cars.Weight <= 4000]
print(cars)

# Q. 5) Instruction ( Applying function on a column ) - Increase all the values of 'MPG_City' column by 3.

def MPG_City(x):
    x = x + 3
    return x

cars['MPG_City'] = cars['MPG_City'].apply(lambda x: MPG_City(x))
print(cars)

# Building a regression equation that predicts the Price (in USD) of a car
# creating a price column
def get_price(y):
    return y.strip('$')

cars['Price'] = cars['MSRP'].map(lambda y: get_price(y))
cars['Price'] = cars['Price'].str.replace(',', '')
cars['Price'].info()
print(cars)


# figuring out the dependent variables
cars = pd.DataFrame(cars, columns=['Price','EngineSize', 'Horsepower', 'MPG_Highway', 'Weight', 'Wheelbase' ])
cars = cars.dropna()
cars['Price'] = pd.to_numeric(cars['Price'])
print(cars.info())
print(cars.corr())
sns.heatmap(cars.corr())
plt.show()


 #Splitting the variable into x and y
y = cars['Price'].astype(int)
x = cars[['EngineSize', 'Horsepower', 'MPG_Highway', 'Weight', 'Wheelbase']]
print(x)
print(y)

# Splitting the dataset into train and test data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
print(x_train.shape, x_test.shape) #Checking the ratio of the train and test variables

#Forming the regression model
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error, r2_score
model = linear_model.LinearRegression()
model.fit(x_train, y_train) #Fitting the model
y_pred = model.predict(x_test)
print(y_pred)
plt.scatter(y_test, y_pred)
plt.show()
print('Coefficients: ', model.coef_)
print('Intercept: ', model.intercept_)
print('MSE: ', r2_score(y_test, y_pred))

