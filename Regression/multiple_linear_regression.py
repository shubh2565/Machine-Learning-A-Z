import pandas as pd 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


#import dataset
data = pd.read_csv('50_Startups.csv')
print(data)
X = data.iloc[ : , : -1].values
y = data.iloc[ : , 4].values
print('\n{}'.format(X))
print('\n{}'.format(y))


#encode text into numbers and one hot encoding for categorical variables
label_X = LabelEncoder()
X[ :, 3] = label_X.fit_transform(X[ : , 3])
encode = OneHotEncoder(categorical_features=[3])
X = encode.fit_transform(X).toarray()
print('\nData after One Hot Encoding:\n{}'.format(X))


#avoiding the multicollinearity
X = X[: , 1:]

#splitting the dataset into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print('\nX-train:\n{}'.format(X_train))
print('\nX-test:\n{}'.format(X_test))
print('\ny-train:\n{}'.format(y_train))
print('\ny-test:\n{}'.format(y_test))


#training and prediction
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_predict = regressor.predict(X_test)
print('\nPredicted Profit          Actual Profit')
for i,j in zip(y_predict, y_test ):
	print('\n{}          {}'.format(i, j))