# shopping

## Introduction

Predicting user behavior is a crucial aspect of e-commerce, enabling businesses to tailor their marketing strategies and improve customer experiences. One key challenge is determining a user's purchasing intent, which can be influenced by various factors such as browsing history, demographics, and device usage. This project leverages machine learning to tackle this problem, developing a classifier that predicts whether a user will make a purchase based on their behavior on a shopping website.

## Overview

This project aims to build a nearest-neighbor classifier to predict a user's purchasing intent. The classifier will be trained on a dataset comprising 12,000 user sessions.

The target variable is the user's purchasing intent (boolean). By analyzing these factors, the classifier will predict the likelihood of a user making a purchase, providing valuable insights for e-commerce businesses to optimize their marketing strategies and enhance customer engagement.

## Implementation


### Classifier

This project utilizes the K-Neighbors Classifier, a supervised learning algorithm that predicts a target variable by finding the most similar instances (nearest neighbors) to a new, unseen instance. In this case, the classifier will identify the k most similar user sessions to a given session, based on features such as pages visited and browser used, and predict the purchasing intent based on the majority vote of these neighbors. The value of k is a hyperparameter that can be tuned for optimal performance. This algorithm is simple yet effective, making it a suitable choice for this classification task.

### Dataset

There are about 12,000 user sessions represented in this spreadsheet: represented as one row for each user session. The first six columns measure the different types of pages users have visited in the session: 

- The `Administrative`, `Informational`, and `ProductRelated` columns measure how many of those types of pages the user visited, and their `corresponding _Duration` columns measure how much time the user spent on any of those pages. 
- The `BounceRates`, `ExitRates`, and `PageValues` columns measure information from Google Analytics about the page the user visited. 
- `SpecialDay` is a value that measures how close the date of the user’s session is to a special day (like Valentine’s Day or Mother’s Day). 
- `Month` is an abbreviation of the month the user visited.
- `OperatingSystems`, `Browser`, `Region`, and `TrafficType` are all integers describing information about the user themself. 
- `VisitorType` will take on the value `Returning_Visitor` for returning visitors and some other string value for non-returning visitors.
- `Weekend` is TRUE or FALSE depending on whether or not the user is visiting on a weekend.
- the `Revenue` column indicates whether the user ultimately made a purchase or not: `TRUE` if they did, `FALSE` if they didn’t.

### How It Works

The general flow of this project is as followed -

1. The `main` function loads data from a CSV spreadsheet by calling the `load_data` function and splits the data into a training and testing set. 
2. The `train_model` function is then called to train a machine learning model on the training data. Then, the model is used to make predictions on the testing data set. 
3. Finally, the `evaluate` function determines the sensitivity and specificity of the model, before the results are ultimately printed to the terminal.

## Usage

To run the project, execute the following command:
```
python shopping.py data
```

## Dependencies

This project relies on the following libraries:

- `train_test_split` from `sklearn.model_selection`.
- `KNeighborsClassifier` from `sklearn.neighbors`.

(If you have scikit.learn, then you are good for the most part)