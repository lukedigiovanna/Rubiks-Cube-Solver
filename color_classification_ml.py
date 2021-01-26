from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense
import os

def load_data(filename):
    data = read_csv(filename, header=1)
    dataset = data.values
    X = dataset[:,:-1]
    y = dataset[:,-1]
    X = X.astype(float)
    y = y.reshape((len(y),1))
    return X, y

EXECUTION_PATH = os.getcwd()

X, y = load_data(os.path.join(EXECUTION_PATH,"color_averages.csv"))

# one hot encode the color values
ohe = OneHotEncoder()
ohe.fit(y)
y = ohe.transform(y).toarray()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33333, random_state=3)

model = Sequential()
model.add(Dense(12, input_dim=3, activation="relu"))
model.add(Dense(8,activation="relu"))
model.add(Dense(6,activation="softmax"))

model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])

model.fit(X_train,y_train,epochs=100,batch_size=10)

_, accuracy = model.evaluate(X_test,y_test)
print(accuracy*100)

model.save("colorclassification.h5")

