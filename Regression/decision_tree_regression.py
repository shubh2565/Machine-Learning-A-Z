import pandas as pd 
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


#import dataset
data = pd.read_csv('Position_Salaries.csv')
print(data)
X = data.iloc[ : , 1:2].values
y = data.iloc[ : , 2:3].values
print('\n{}'.format(X))
print('\n{}'.format(y))


#fitting the model to the dateset and predicting
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)
print('\nSalary predicted at level 6.5 : {}'.format(regressor.predict(6.5)))


#visualize
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('Decision Tree Regression Model')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()