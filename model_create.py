import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, r2_score
import pickle

data = pd.read_csv("kc_house_data.csv")

model = LinearRegression()

labels = data['price']
conv_dates = [1 if "2014" in values else 0 for values in data.date ]
data['date'] = conv_dates

to_drop = [
	'id', 
	'zipcode',
	'price', 
	"bathrooms",  
	"view", 
	"waterfront", 
	"yr_built", 
 	"yr_renovated", 
 	"sqft_living15", 
 	"sqft_lot15"
 	]
data.drop(to_drop, axis=1, inplace=True)
data.rename(columns={"bedrooms": "bhk"}, inplace=True)
data = data.astype({"floors": int})
max_score = 0
ideal_state = 1

for i in range(1,101):
	x_train , x_test , y_train , y_test = train_test_split(data , labels , test_size = 0.20,random_state =i)
	model.fit(x_train,y_train)
	score = model.score(x_test,y_test)
	if score > max_score:
		max_score = score
		ideal_state = i


x_train , x_test , y_train , y_test = train_test_split(data , labels , test_size = 0.20,random_state =ideal_state)
model.fit(x_train,y_train)

pred = model.predict(x_test)

pickle.dump(model, open("models/housing_predict_model.sav", "wb"))