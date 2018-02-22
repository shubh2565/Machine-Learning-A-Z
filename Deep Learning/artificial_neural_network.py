import pandas as pd 
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix




#import the dataset
data = pd.read_csv('Churn_Modelling.csv')
print(data)
X = data.iloc[ : , 3:13].values
y = data.iloc[ : , 13].values
#print('\n{}'.format(X))
#print('\n{}'.format(y))


#encode text into numbers and one hot encoding for categorical variables
label_X1 = LabelEncoder()
label_X2 = LabelEncoder()
X[ :, 1] = label_X1.fit_transform(X[ : , 1])
X[ :, 2] = label_X2.fit_transform(X[ : , 2])
encode = OneHotEncoder(categorical_features=[1])
X = encode.fit_transform(X).toarray()
X = X[ : , 1:]
print('\nData after One Hot Encoding:\n{}'.format(X))


##splitting the dataset into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print('\nX-train:\n{}'.format(X_train))
print('\nX-test:\n{}'.format(X_test))
print('\ny-train:\n{}'.format(y_train))
print('\ny-test:\n{}'.format(y_test))


# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
print('\nX-train after feature scaling:\n{}'.format(X_train))
print('\nX-test after feature scaling:\n{}'.format(X_test))


#initializing the ANN and adding input and first hidden layer
classifier = Sequential()
classifier.add(Dense(units=6, kernel_initializer='glorot_uniform',activation='relu',input_dim=11))

#adding second and third hidden layer
classifier.add(Dense(units=6, kernel_initializer='glorot_uniform',activation='relu'))

#adding output layer
classifier.add(Dense(units=1, kernel_initializer='glorot_uniform',activation='sigmoid'))

#compiling the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#fitting the ANN to traing set
classifier.fit(X_train, y_train, batch_size=32, epochs=100)

#prediction
y_pred = classifier.predict(X_test)
y_pred =(y_pred > 0.5)

#confusion matrix
cm = confusion_matrix(y_test, y_pred)
print('\n\nConfusion Matrix:\n{}\n\n'.format(cm))