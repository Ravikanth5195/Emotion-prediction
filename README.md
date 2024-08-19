# Emotion Prediction Project

This project focuses on classifying emotions in text data using machine learning models. Three different algorithms were implemented and compared: Naive Bayes, Neural Networks, and Random Forest. The goal was to build a model that accurately predicts the emotion expressed in a given text.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Models Implemented](#models-implemented)
  - [Naive Bayes](#naive-bayes)
  - [Neural Networks](#neural-networks)
  - [Random Forest](#random-forest)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Overview

Emotion classification is an important task in natural language processing (NLP) with applications in sentiment analysis, chatbots, social media monitoring, and more. This project aims to classify text into one of several emotions such as happiness, sadness, anger, fear, etc.

## Dataset

The project uses the Emotion Dataset from Kaggle, which contains text data labeled with various emotions. The dataset was preprocessed to clean the text, remove stopwords, and convert the text into a numerical format suitable for machine learning algorithms.

## Models Implemented

### Naive Bayes

- **Overview**: A probabilistic model that applies Bayes' Theorem with strong (naive) independence assumptions between the features.
- **Implementation**: Multinomial Naive Bayes was used, which is suitable for text classification tasks.
- **Performance**: This model performed well on the dataset, providing a good balance between accuracy and interpretability.

### Neural Networks

- **Overview**: A deep learning model designed to capture complex patterns in data.
- **Implementation**: A sequential model with multiple dense layers and dropout was implemented using TensorFlow/Keras.
- **Performance**: Unfortunately, the neural network did not yield the expected accuracy, likely due to the limited size of the dataset and the relatively simple architecture. Further tuning and more complex architectures might improve performance, but this was not the focus of this project.

### Random Forest

- **Overview**: An ensemble learning method that builds multiple decision trees and merges them together to get a more accurate and stable prediction.
- **Implementation**: A Random Forest Classifier with 100 estimators was used.
- **Performance**: This model performed better than the neural network and provided robust accuracy, making it a strong candidate for emotion classification tasks.

## Results

- **Naive Bayes**: Achieved good accuracy and was computationally efficient.
- **Neural Networks**: Did not perform as well as expected, with lower accuracy compared to other models.
- **Random Forest**: Provided the best overall performance with high accuracy and stability.

## Installation

To run this project, you need to have Python installed along with the following libraries:

```bash
pip install pandas numpy nltk scikit-learn tensorflow matplotlib xgboost
