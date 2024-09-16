import csv
import sys
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

TEST_SIZE = 0.4
EPOCHS = 10

months = {
    "Jan": 0,
    "Feb": 1,
    "Mar": 2,
    "Apr":3,
    "May":4,
    "June":5,
    "Jul":6,
    "Aug":7,
    "Sep":8,
    "Oct":9,
    "Nov":10,
    "Dec":11
}

if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

evidence = []
label = []
integers = [0, 2, 4, 10, 11, 12, 13, 14, 15, 16]
floats = [1, 3, 5, 6, 7, 8, 9]

with open("shopping.csv") as file:
    reader = csv.reader(file)
    header = next(reader)

    for row in reader:
        month_name = row[10]
        row[10] = months[month_name]
        

        row[15] = (1 if row[15] == "Returning_Visitor" else 0)
        
        weekend = (1 if row[16] == "TRUE" else 0)
        row[16] = weekend

        purchase = (0 if row[len(row)-1] == "FALSE" else 1)


        for i in integers:
            row[i] = int(row[i])

        for j in floats:
            row[j] = float(row[j])
        
        row.remove(row[17])

        evidence.append(row)
        label.append(purchase)


X_train, X_test, y_train, y_test = train_test_split(
        np.array(evidence), np.array(label), test_size=TEST_SIZE
    )

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.fit(X_train, y_train, epochs=EPOCHS)

model.evaluate(X_test, y_test, verbose=2)