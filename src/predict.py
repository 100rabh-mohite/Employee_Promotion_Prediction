import joblib

model=joblib.load("model/model.pkl")

years=[[6]]

prediction=model.predict(years)

if prediction[0]==1:
    print("promoted")
else:
    print("not promoted")