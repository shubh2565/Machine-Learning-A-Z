import pandas as pd 
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


#import dataset
data = pd.read_csv('Position_Salaries.csv')
print(data)
X = data.iloc[ : , 1:2].values
y = data.iloc[ : , 2].values
print('\n{}'.format(X))
print('\n{}'.format(y))


#traing and visualization
poly_reg = PolynomialFeatures(degree=2)
X_poly = poly_reg.fit_transform(X)
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)


#predict
print('Salary predicted at level 6.5 : {}'.format(lin_reg.predict(poly_reg.fit_transform(6.5))))


#visualize
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, lin_reg.predict(poly_reg.fit_transform(X_grid)), color='blue')
plt.title('Polynomial Regression Model')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()