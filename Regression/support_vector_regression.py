import pandas as pd 
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


#import dataset
data = pd.read_csv('Position_Salaries.csv')
print(data)
X = data.iloc[ : , 1:2].values
y = data.iloc[ : , 2:3].values
print('\n{}'.format(X))
print('\n{}'.format(y))


#feature scaling so that no variable dominates over the other(s)
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)
print('\nX after feature scaling:\n{}'.format(X))
print('\ny after feature scaling:\n{}'.format(y))


#fitting and predicting SVR to the dataset
regressor = SVR(kernel = 'rbf')
regressor.fit(X, y)
y_pred = regressor.predict(sc_X.transform(np.array([[6.5]])))
print('Salary predicted at level 6.5 : {}'.format(sc_y.inverse_transform(y_pred)))


#visualize
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('Support Vector Regression Model')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()