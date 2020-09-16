# Naive-Bayes
Naive Bayes algorithm for prediction in machine learning

The goal of the project is to present an implementation of a machine learning using classification model to predict the outcome of events based on naive Bayes probability.
The code trains a model to predict in which class the given input data belongs to, in another words, given the input the model predicts the most probable outcome.
The model is prepared to taken all features as numbers and the classes (outcome) as string, the data file used to trains the model (players_stats.csv) is attached in the repository, it is a template for data file to train the model based on any data set with variable number of features and classes.

The concrete case of the data file, the model predicts the winner of best players award based on his perfomance
features are: 

[Goals, Assists, Games Played, trophies, First/Second/Third]

To test the model having the training data set, provide only the features in the vector input. The source of data to train the model I credit: https://www.mockaroo.com​.
The source code was originally written in MAC OS Catalina using python 3, in order to compile after downloding all the files from this repository, type:
> python3 main.py
