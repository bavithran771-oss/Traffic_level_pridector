import pandas as pd
from flask import Flask, render_template, request
from sklearn.tree import DecisionTreeClassifier

app = Flask(__name__)
data = pd.read_csv("traffic_data.csv")
features = ["hour", "day"]
X = pd.get_dummies(data[features])
y = data["traffic_level"]
model = DecisionTreeClassifier(max_depth=4, random_state=2)
model.fit(X, y)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    shown_time = None
    if request.method == "POST":
        selected_time = request.form["time"]
        hour = int(selected_time.split(":")[0])
        hour_12 = hour % 12 or 12
        period = "AM" if hour < 12 else "PM"
        shown_time = f"{hour_12}:{selected_time.split(":")[1]} {period}"
        values = pd.DataFrame([[hour, request.form["day"]]], columns=features)
        values = pd.get_dummies(values).reindex(columns=X.columns, fill_value=0)
        prediction = model.predict(values)[0]
    return render_template("index.html", prediction=prediction, shown_time=shown_time)

if __name__ == "__main__":
    app.run(debug=True)
