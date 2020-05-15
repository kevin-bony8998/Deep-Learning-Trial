from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import statistics
import keras
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense

def error_finder():
	model.compile(optimizer = 'adam', loss = 'mean_squared_error')
	model.fit(X_train, y_train,epochs=50)
	predictions = model.predict(X_test)
	return (mean_squared_error(y_test,predictions))

loss=[]
model = Sequential()
concrete_data = pd.read_csv("C:/Users/KEVINBONYTHEKKANATH-/Desktop/Zelish/Codes/concrete_data.csv")
print(concrete_data.head())
concrete_data_columns = concrete_data.columns

predictors = concrete_data[concrete_data_columns[concrete_data_columns != 'Strength']] # all columns except Strength
target = concrete_data['Strength'] # Strength column
predictors_norm = (predictors - predictors.mean()) / predictors.std()

X_train, X_test, y_train, y_test = train_test_split(predictors_norm, target, test_size=0.3, random_state=42)
n_cols = predictors_norm.shape[1]
print(predictors_norm.shape[1])

model.add(Dense(10, activation = 'relu', input_shape = (n_cols,)))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(1, activation = 'relu'))

for i in range(0,50):
	loss.append(error_finder())
	print(loss)
print("Mean of errors: "+str(statistics.mean(loss)))
print("Standard Deviation of errors: "+str(statistics.stdev(loss)))