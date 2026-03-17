from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Train model when app starts
data = pd.read_csv("dataset.csv")
X = data.drop("disease", axis=1)
y = data["disease"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        fever    = int(request.form.get("fever", 0))
        cough    = int(request.form.get("cough", 0))
        fatigue  = int(request.form.get("fatigue", 0))
        headache = int(request.form.get("headache", 0))
        nausea   = int(request.form.get("nausea", 0))
        input_data = pd.DataFrame([[fever, cough, fatigue, headache, nausea]],
                    columns=["fever", "cough", "fatigue", "headache", "nausea"])
        prediction = model.predict(input_data)[0]
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)