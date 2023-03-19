import pandas as pd

from sklearn.linear_model import LinearRegression


data = pd.read_csv("api/service/population_visitors.csv")
X = data["Population"].values.reshape(-1, 1)
y = data["Visitors"].values.reshape(-1, 1)

# Create a LinearRegression model and fit it to the data
model = LinearRegression()
model.fit(X, y)


def predict_visitors_with_populatuion(population):
    prediction = model.predict([[int(population)]])
    print({"pop": int(population), "vis": int(prediction[0][0])})
    return int(prediction[0][0])
