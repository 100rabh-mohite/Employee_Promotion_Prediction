import joblib

def test_Employee_prediction():
    model=joblib.load("model/model.pkl")

    prediction=model.predict([[6]])

    assert prediction[0]==1