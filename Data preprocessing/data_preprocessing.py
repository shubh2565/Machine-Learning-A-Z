import pandas as pd 
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split


#loading of dataset
data = pd.read_csv('Data.csv')
print(data)
X = data.iloc[ : , : -1].values
y = data.iloc[ : , 3].values
print('\n{}'.format(X))
print('\n{}'.format(y))


#replacing missing data
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer = imputer.fit(X[ : , 1:3])
X[ : , 1:3] = imputer.transform(X[ : , 1:3])
print('\n{}'.format(X))


#encode text into numbers and one hot encoding for categorical variables
label_X = LabelEncoder()
X[ :, 0] = label_X.fit_transform(X[ : , 0])
encode = OneHotEncoder(categorical_features=[0])
X = encode.fit_transform(X).toarray()
print('\nData after One Hot Encoding:\n{}'.format(X))

label_y = LabelEncoder()
y = label_y.fit_transform(y)
print('\n{}'.format(y))


#splitting the dataset into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print('\nX-train:\n{}'.format(X_train))
print('\nX-test:\n{}'.format(X_test))
print('\ny-train:\n{}'.format(y_train))
print('\ny-test:\n{}'.format(y_test))


#feature scaling so that no variable dominates over the other(s)
feature = StandardScaler()
X_train = feature.fit_transform(X_train)
X_test = feature.transform(X_test)
print('\nX-train after feature scaling:\n{}'.format(X_train))
print('\nX-test after feature scaling:\n{}'.format(X_test))