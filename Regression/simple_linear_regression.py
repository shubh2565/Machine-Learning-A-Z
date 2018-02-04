import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


#import dataset and split it into training and test set
data = pd.read_csv('Salary_Data.csv')
print(data)
X = data.iloc[ : , : -1].values
y = data.iloc[ : , 1].values
print('\n{}'.format(X))
print('\n{}'.format(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)
print('\nX-train:\n{}'.format(X_train))
print('\nX-test:\n{}'.format(X_test))
print('\ny-train:\n{}'.format(y_train))
print('\ny-test:\n{}'.format(y_test))


#training and prediction
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_predict = regressor.predict(X_test)
print('\nExperience    Predicted Salaries    Actual Salaries')
for i,j,k in zip(X_test, y_predict, y_test ):
	print('\n{}        {}       {}'.format(i, j, k))


#visualizing the results
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary VS Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary VS Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()