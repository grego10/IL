from bs4 import BeautifulSoup
from sklearn.linear_model import LinearRegression
from dotenv import load_dotenv

import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import requests
import pickle


def raw_museum():
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
    with open("1_raw_museum.html", "w") as f:
        f.write(html)

    f.close()


def intermediate_museum():
    with open("1_raw_museum.html") as html:
        soup = BeautifulSoup(html, "html.parser")

    table_rows = soup.find_all("tr")
    html.close()
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
    df.to_csv("2_intermediate_museums.csv", index=False)


def raw_cities():
    load_dotenv()
    df = pd.read_csv("2_intermediate_museums.csv")

    cities = df["City"].unique()

    # Create a new dataframe with the unique cities
    city_df = pd.DataFrame({"City": cities})

    city_df["Population"] = np.nan
    print(city_df)
    for index, row in city_df.iterrows():
        name = row[0].split(",")[0]
        api_url = "https://api.api-ninjas.com/v1/city?name={}".format(name)
        response = requests.get(api_url, headers={"X-Api-Key": os.getenv("API_NINJA")})
        res = json.loads(response.text)
        print(response.status_code)
        print(res)
        if response.status_code == 200 and len(res) > 0 and res[0]["name"] == name:
            city_df.loc[city_df["City"] == row[0], "Population"] = res[0]["population"]
        else:
            print("error: ", name)

    city_df.to_csv("1_raw_cities.csv", index=False)


def primary_museum_cities():
    df_museum = pd.read_csv("2_intermediate_museums.csv")
    df_cities = pd.read_csv("1_raw_cities.csv")

    merged_data = pd.merge(df_museum, df_cities, on="City")

    merged_data[["Museum", "City", "Population", "Visitors"]].to_csv(
        "3_primary_museum_city.csv", index=False
    )


def feature_visitor_population():
    df = pd.read_csv("3_primary_museum_city.csv")
    # Filter the dataframe to keep only museums with more than 2,000,000 visitors
    df_filtered = df[df["Visitors"] > 2000000]
    df_filtered.to_csv("4_feature_visitor_population.csv", index=False)


def model_visitor_population():
    data = pd.read_csv("4_feature_visitor_population.csv")
    X = data["Population"].values.reshape(-1, 1)
    y = data["Visitors"].values.reshape(-1, 1)

    # Create a LinearRegression model and fit it to the data
    model = LinearRegression()
    model.fit(X, y)
    pickle.dump(model, open("5_model_linear_regression", "wb"))


def output_visitors_pop(pop):
    loaded_model = pickle.load(open("5_model_linear_regression", "rb"))
    prediction = int(loaded_model.predict([[pop]])[0][0])
    print(prediction)
    return prediction


def output_visitors_pop_csv():
    data = pd.read_csv("population_visitors.csv")

    # Extract the features (population) and target (visitors) variables
    X = data["Population"].values.reshape(-1, 1)

    loaded_model = pickle.load(open("5_model_linear_regression", "rb"))

    # Make predictions based on population
    predictions = loaded_model.predict(X)

    # Add the predicted visitors column to the DataFrame
    data["Predicted_Visitors"] = predictions

    # Save the updated DataFrame to a new CSV file
    data.to_csv("6_output_visitors_pop.csv", index=False)


def bar_chart_actual_vs_predicted_visitors():
    df = pd.read_csv("6_output_visitors_pop.csv")
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
    df = pd.read_csv("6_output_visitors_pop.csv")

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
    # raw_museum()
    # intermediate_museum()
    # primary_museum()
    # raw_cities()
    # primary_museum_cities()
    # feature_visitor_population()
    # model_visitor_population()
    # output_visitors_pop(10000000)
    # output_visitors_pop_csv()
    # bar_chart_actual_vs_predicted_visitors()
    sacatter_plot_actual_vs_predicted_visitors()
