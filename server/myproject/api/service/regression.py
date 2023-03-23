import pandas as pd

import pickle


model = pickle.load(open("api/service/5_model_linear_regression", "rb"))


def predict_visitors_with_population(population):
    prediction = model.predict([[int(population)]])
    print({"pop": int(population), "vis": int(prediction[0][0])})
    return int(prediction[0][0])
