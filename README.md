
# Intent Analysis using Random Forests

## Introduction

Predicting user behavior is pivotal in the realm of e-commerce, as it allows businesses to refine their marketing strategies and enhance customer experiences. Understanding a user's purchasing intent involves analyzing a myriad of factors including browsing history, demographics, device usage, and more.

This project addresses this challenge by employing an artificial neural network (ANN) to predict a user's likelihood of making a purchase. By leveraging the power of neural networks, we can capture complex patterns and interactions within behavioral data, providing more accurate predictions compared to traditional methods. The model is designed to analyze various inputs such as time of day, day of the week, date, holiday status, shopping duration, and other behavioral metrics to determine the probability of a purchase.
## Overview

This project aims to build a Artificial Neural Network to predict a user's purchasing intent. The network will be trained on a dataset comprising 12,000 user sessions.

The target variable is the user's purchasing intent (boolean). By analyzing these factors, the network will predict the likelihood of a user making a purchase, providing valuable insights for e-commerce businesses to optimize their marketing strategies and enhance customer engagement.

## Implementation

### Model

This project utilizes a Neural Network, a supervised learning algorithm that predicts a target variable by learning complex patterns in the data through layers of interconnected nodes (neurons). In this case, the neural network analyzes user sessions, using features such as pages visited, browser used, time spent, and other behavioral data, to predict purchasing intent. The network adjusts its internal parameters (weights) during training to minimize prediction error. 
This architecture allows for greater flexibility and accuracy compared to simpler models like K-Neighbors, as it can capture non-linear relationships in the data. Hyperparameters such as the number of layers, neurons per layer, activation functions, and learning rate can be fine-tuned to achieve optimal performance. 
*Neural networks are particularly effective for this classification task, given the complexity of user behavior patterns.*

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

You can experiment with the following ones - 
- **`TEST_SIZE`** represents the portion of dataset which will be used for testing (eg, 0.2 = 20% of the data).
- **`EPOCHS`** represents the number of times the network will run through the training dataset.

This dataset is provided by [Kaggle](https://www.kaggle.com/datasets/imakash3011/online-shoppers-purchasing-intention-dataset)

### Project Workflow

1. The `main` function loads data from a CSV file by calling the `load_data` function. The dataset is then preprocessed and split into training and testing sets. This ensures the model can learn from one portion of the data and be evaluated on unseen data.
    
2. **Model Training**:  
    The `train_model` function is invoked to build and train an Artificial Neural Network (ANN) on the training data. The network's architecture, including the number of layers and neurons, is defined in this step, and the model learns by adjusting weights through backpropagation. After training, the model is used to make predictions on the test set.
    
3. **Evaluation**:  
    The `evaluate` function is responsible for assessing the performance of the model by calculating key metrics like sensitivity and specificity. These metrics help determine the model’s ability to correctly identify positive and negative classes (e.g., predicting user purchase intent). The final evaluation results are printed to the terminal for review.

## Output

The training process of the neural network generates several metrics at each epoch, providing insights into how well the model is learning. Below is an explanation of the key components seen in the output:

- **Epoch X/Y**: Indicates the current epoch number (X) out of the total number of epochs (Y) the model will run during training. Each epoch represents one complete pass through the training dataset.
    
- **Steps per Epoch**: Displays the number of batches processed in each epoch (e.g., `232/232` means all 232 batches have been processed). This is based on the size of the dataset and the batch size.
    
- **Accuracy**: Shows the percentage of correctly predicted outputs out of the total predictions made during that epoch. The accuracy metric is crucial for assessing how well the model is performing on the training data. Higher values indicate better performance.
    
- **Loss**: Reflects how far the model’s predictions are from the actual targets. Lower values represent better model performance. The loss is a key measure during training as it guides the model's weight updates through backpropagation.
    
- **Step Time**: Indicates the time taken to process each step in the epoch, usually in milliseconds or microseconds (e.g., `781us/step`). This gives a sense of the training speed.
    
- **Validation/Test Accuracy and Loss**: After training is complete, the model is evaluated on a separate validation or test dataset. The output will display the final accuracy and loss for this unseen data (e.g., `accuracy: 0.8889`, `loss: 0.3981`). These metrics are important because they show how well the model generalizes to data it hasn't seen before.

## Usage

To run the project, execute the following command:
```
python shopping_nn.py data
```

## Dependencies

This project relies on the following libraries:

- **csv**: Handles reading and writing CSV files (Python standard library).
- **sys**: Manages system-specific parameters and command-line arguments (Python standard library).
- **NumPy** (`numpy`): Supports large arrays and mathematical operations.  
    Install: `pip install numpy`
- **TensorFlow** (`tensorflow`): Builds and trains the neural network model.  
    Install: `pip install tensorflow`
- **Scikit-learn** (`sklearn`): Provides tools for data splitting and evaluation.  
    Install: `pip install scikit-learn`
