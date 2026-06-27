from flask import Flask

import joblib

app=Flask(__name__)

model=joblib.load("model/model.pkl")

@app.route("/")

def home():
    prediction=model.predict([[6]])
    result= "promoted" if prediction[0]==1 else "not promoted"

    return f"Employee result: {result}"


if __name__=="__main__":
    app.run(debug=True)
