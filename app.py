from flask import Flask, render_template, request
import joblib


app = Flask(__name__)

# Load model & vectorizer
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = ""
    if request.method == "POST":
        message = request.form["message"]
        data = vectorizer.transform([message])
        result = model.predict(data)[0]
        prediction = "🚨 Spam" if result == "spam" else "✅ Ham"
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
