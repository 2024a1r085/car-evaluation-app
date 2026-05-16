from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model and encoders
model = joblib.load("model.pkl")
encoders = joblib.load("encoders.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = ""
    features = {}

    if request.method == "POST":

        buying = request.form["buying"]
        maint = request.form["maint"]
        doors = request.form["doors"]
        persons = request.form["persons"]
        lug_boot = request.form["lug_boot"]
        safety = request.form["safety"]

        # Encode inputs
        input_data = [
            encoders["buying"].transform([buying])[0],
            encoders["maint"].transform([maint])[0],
            encoders["doors"].transform([doors])[0],
            encoders["persons"].transform([persons])[0],
            encoders["lug_boot"].transform([lug_boot])[0],
            encoders["safety"].transform([safety])[0]
        ]

        input_array = np.array([input_data])

        pred = model.predict(input_array)[0]

        prediction = encoders["class"].inverse_transform([pred])[0]

        features = {
            "Buying Price":       buying,
            "Maintenance Cost":   maint,
            "Number of Doors":    doors,
            "Passenger Capacity": persons,
            "Luggage Boot Size":  lug_boot,
            "Safety Rating":      safety,
        }

    return render_template("index.html", prediction=prediction, features=features)

if __name__ == "__main__":
    app.run(debug=True)
