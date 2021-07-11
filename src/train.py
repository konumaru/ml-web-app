import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def main():
    data = pd.read_csv("../data/ufos.csv")

    # Clean data.
    data = data.dropna()
    data = data[(data["duration (seconds)"] >= 1) & (data["duration (seconds)"] <= 60)]
    data["country"] = LabelEncoder().fit_transform(data["country"])

    # Train model.
    selected_features = ["duration (seconds)", "latitude", "longitude"]
    label_feature = "country"

    X = data[selected_features]
    y = data[label_feature]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    model = LogisticRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print(classification_report(y_test, predictions))
    print("predict labels: ", predictions)
    print("Accuracy: ", accuracy_score(y_test, predictions))

    # Save model.
    model_filename = "../data/ufo-model.pkl"
    pickle.dump(model, open(model_filename, "wb"))

    # test saved model.
    model = pickle.load(open(model_filename, "rb"))
    print(model.predict([[50, 44, -12]]))


if __name__ == "__main__":
    main()
