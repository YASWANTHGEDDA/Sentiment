from flask import Flask, render_template, request
import pickle
import os

app = Flask(__name__)

# Load model and vectorizer
model_path = "model/sentiment_model.pkl"
vectorizer_path = "model/vectorizer.pkl"

model = pickle.load(open(model_path, "rb"))
vectorizer = pickle.load(open(vectorizer_path, "rb"))

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        user_input = request.form["text"]
        if user_input.strip() == "":
            prediction = "Please enter some text."
        else:
            vectorized = vectorizer.transform([user_input])
            prediction = model.predict(vectorized)[0]
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
