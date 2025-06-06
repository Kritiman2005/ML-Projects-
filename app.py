import pickle
from flask import Flask,render_template,request
app = Flask(__name__)

# Correctly load the model and vectorizer
tokenizer = pickle.load(open('cv.pkl', 'rb'))
model = pickle.load(open('clf.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index2.html')

@app.route('/predict', methods=['POST'])
def predict():
    email = request.form.get('content')
    if email:
        tokenized = tokenizer.transform([email])
        prediction = model.predict(tokenized)[0]  # Get the first value
        prediction = 1 if prediction == 1 else -1
        return render_template("index2.html", prediction=prediction, email=email)
    else:
        return render_template("index2.html", prediction="No content provided", email="")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
 