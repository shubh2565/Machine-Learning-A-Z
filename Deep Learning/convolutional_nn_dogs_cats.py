from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator



classifier = Sequential()

#step1 - Convolution
classifier.add(Convolution2D(32, 3 , 3, input_shape=(64,64,3), activation='relu'))

#step2 - Max Pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))

#repeat to make a stronger classifier
classifier.add(Convolution2D(32, 3 , 3, activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

#step3 - Flattening
classifier.add(Flatten())

#step3 - Fully Connected Layer
classifier.add(Dense(units=128, activation='relu',))
classifier.add(Dense(units=1, activation='sigmoid'))

#compiling the CNN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#image preprocessing
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set', target_size=(64, 64), batch_size=32, class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set', target_size=(64, 64), batch_size=32, class_mode='binary')

classifier.fit_generator(training_set, steps_per_epoch=8000, epochs=25, validation_data=test_set, validation_steps=2000)

#predict
y = classifier.predict('predict.jpeg')
print(y)