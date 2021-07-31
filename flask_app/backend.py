import pickle
import numpy as np

def conv_numeric(x):
	try:
		return int(x)
	except:
		return float(x)

class house_price:
	def __init__(self):
		with open("../models/housing_predict_model.sav", "rb") as f:
			self.model = pickle.load(f)
	def predict(self, data):
		data = list(map(conv_numeric, data))
		if data[0] == 2014:
			data[0] = 1
		else:
			data[0] = 0
		data = np.array(data)
		data = data.reshape(1, -1)
		pred = self.model.predict(data)[0]
		return round(pred, ndigits=2)