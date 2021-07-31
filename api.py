from flask import (
	Flask, 
	request, 
	render_template, 
	make_response
	)
from backend import house_price

app = Flask(__name__)
house = house_price()

@app.route("/")
def index():
	return "<h1 align='center'>Welcome to Mehedi's Projects</h1>"

@app.route("/house-price", methods=["GET", "POST"])
def price():
	if request.method == "GET":
		return render_template("prediction.html", price=False)
	data = request.form.to_dict()
	get_price = house.predict(data.values())

	return render_template("prediction.html", price=get_price)

if __name__ == "__main__":
	app.run(debug=True)