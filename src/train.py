import numpy as np
from sklearn.tree import DecisionTreeClassifier
import joblib
import os

# employee working experience in years

X=np.array([
    [1],
    [2],
    [3],
    [4],
    [5],
    [6],
    [7],
    [8]
])


# 0= not promoted, 1= promoted

y=np.array([
    0,
    0,
    0,
    0,
    1,
    1,
    1,
    1
])

model=DecisionTreeClassifier()
model.fit(X,y)

os.makedirs("model",exist_ok=True)

joblib.dump(model,"model/model.pkl")
print("Model Trained Successfully")
