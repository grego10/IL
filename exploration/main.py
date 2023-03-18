import json
import requests
from bs4 import BeautifulSoup

import pandas as pd

import os

from dotenv import load_dotenv

import numpy as np
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt


def extract_museum():
    # Set the URL for the Wikipedia API
    url = "https://en.wikipedia.org/w/api.php"

    # Set the parameters for the API request
    params = {
        "action": "parse",
        "page": "List_of_most-visited_museums",
        "prop": "text",
        "format": "json",
        "section": 4,
    }

    # Send the API request and get the response
    response = requests.get(url, params=params).json()
    html = response["parse"]["text"]["*"]

    soup = BeautifulSoup(html, "html.parser")
    table_rows = soup.find_all("tr")

    # Loop through each table row nad parse it
    output = []
    for row in table_rows:

        row = row.find_all("td")
        parsed_row = [val.text.strip() for val in row]
        if len(parsed_row) != 3:
            continue

        parsed_row_as_dict = {
            "Museum": parsed_row[0],
            "City": parsed_row[1],
            "Visitors": parsed_row[2],
        }
        output.append(parsed_row_as_dict)

    # Clean the data
    df = pd.DataFrame(output)
    df["Visitors"] = df["Visitors"].str.replace(r"\[.*\]", "", regex=True)
    df["Visitors"] = df["Visitors"].str.replace(",", "").astype(int)
    df.to_csv("museums.csv", index=False)


def filter_museum_and_get_cities():
    df = pd.read_csv("museums.csv")
    # filter the dataframe to keep only museums with more than 2,000,000 visitors
    df_filtered = df[df["Visitors"] > 2000000]
    df_filtered.to_csv("museum_filtered.csv", index=False)

    cities = df_filtered["City"].unique()

    # create a new dataframe with the unique cities
    city_df = pd.DataFrame({"City": cities})

    # print the city dataframe
    city_df.to_csv("cities.csv", index=False)


def get_city_population():
    load_dotenv()
    os.getenv("API_NINJA")

    df = pd.read_csv("cities.csv")
    df["Population"] = np.nan
    for index, row in df.iterrows():
        name = row[0].split(",")[0]
        api_url = "https://api.api-ninjas.com/v1/city?name={}".format(name)
        response = requests.get(api_url, headers={"X-Api-Key": os.getenv("API_NINJA")})
        res = json.loads(response.text)
        if response.status_code == requests.codes.ok and res[0]["name"] == name:
            df.loc[df["City"] == row[0], "Population"] = res[0]["population"]
        else:
            print("Error", name)

    df.to_csv("cities_population.csv", index=False)


def get_visitors_per_population():
    museum_data = pd.read_csv("museum_filtered.csv")
    population_data = pd.read_csv("cities_population.csv")

    merged_data = pd.merge(museum_data, population_data, on="City")

    merged_data[["Museum", "City", "Population", "Visitors"]].to_csv(
        "population_visitors.csv", index=False
    )


def predict_visitors_with_populatuion(populaton):
    data = pd.read_csv("population_visitors.csv")
    X = data["Population"].values.reshape(-1, 1)
    y = data["Visitors"].values.reshape(-1, 1)

    # Create a LinearRegression model and fit it to the data
    model = LinearRegression()
    model.fit(X, y)

    # Make a prediction for a city with a population of 10 million
    prediction = model.predict([[populaton]])
    print(prediction)


def predict_visitors_csv():
    data = pd.read_csv("population_visitors.csv")

    # Extract the features (population) and target (visitors) variables
    X = data["Population"].values.reshape(-1, 1)
    y = data["Visitors"].values.reshape(-1, 1)

    # Create a LinearRegression model and fit it to the data
    model = LinearRegression()
    model.fit(X, y)

    # Make predictions based on population
    predictions = model.predict(X)

    # Add the predicted visitors column to the DataFrame
    data["Predicted_Visitors"] = predictions

    # Save the updated DataFrame to a new CSV file
    data.to_csv("predicted_visitors.csv", index=False)


def bar_chart_actual_vs_predicted_visitors():
    df = pd.read_csv("predicted_visitors.csv")
    # create a figure and axis object
    fig, ax = plt.subplots(figsize=(20, 6))

    # plot the actual visitors and predicted visitors as two sets of bars
    bar_width = 0.35
    opacity = 0.8
    ax.bar(
        df.index,
        df["Visitors"],
        bar_width,
        alpha=opacity,
        color="b",
        label="Actual Visitors",
    )
    ax.bar(
        df.index + bar_width,
        df["Predicted_Visitors"],
        bar_width,
        alpha=opacity,
        color="r",
        label="Predicted Visitors",
    )

    # add labels and titles
    ax.set_xlabel("Museum", fontsize=10)
    ax.set_ylabel("Number of Visitors", fontsize=10)
    ax.set_title("Actual Visitors vs. Predicted Visitors")
    ax.set_xticks(df.index + bar_width / 2)
    ax.set_xticklabels(
        [museum.replace(" ", "\n") for museum in df["Museum"]], fontsize=10
    )
    ax.legend()

    # display the plot
    plt.show()


def sacatter_plot_actual_vs_predicted_visitors():
    df = pd.read_csv("predicted_visitors.csv")

    fig, ax = plt.subplots(figsize=(10, 6))

    # plot the actual visitors and predicted visitors as scatter points
    ax.scatter(df["Population"], df["Visitors"], color="b", label="Actual Visitors")
    ax.scatter(
        df["Population"],
        df["Predicted_Visitors"],
        color="r",
        label="Predicted Visitors",
    )

    # add labels and titles with reduced font size
    ax.set_xlabel("Population", fontsize=12)
    ax.set_ylabel("Number of Visitors", fontsize=12)
    ax.set_title("Population vs. Visitors", fontsize=14)
    ax.legend(fontsize=12)

    # display the plot
    plt.show()


if __name__ == "__main__":
    # extract_museum()
    # filter_museum_and_get_cities()
    # get_city_population()
    # get_visitors_per_population()
    # predict_visitors_with_populatuion(1000000)
    # predict_visitors_csv()
    # bar_chart_actual_vs_predicted_visitors()
    sacatter_plot_actual_vs_predicted_visitors()
