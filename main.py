from csv import reader
from math import sqrt
from math import pi
from math import exp
from random import randrange

# Function Purpose: Open the CSV data file
# @input: file name
# @output: The list with all the row of dataset
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue  # skip empty rows with no data
            dataset.append(row)
    return dataset

# Function Purpose: Convert column data into double values
# @input: data set, specific column
# @output: None
def parse_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())

# Function Purpose: Convert column with outcome name to the correspondent int value
# @input: data set, specific column
# @output: A dictionary with column name (key) and int value (value)
def parse_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique_key = set(class_values)
    column_name_and_value_dict = dict()
    print("---------------------------------------------------")
    print("Outcome Label \t\t Value Label")
    print("---------------------------------------------------")
    for i, value in enumerate(unique_key):
        column_name_and_value_dict[value] = i
        print(value,"\t\t", i)
    for row in dataset:
        row[column] = column_name_and_value_dict[row[column]]
    return column_name_and_value_dict

# Function Purpose: Split the input data into training and test data set
# @input: Data set, split ratio
# @output: training set, test set
def split_dataset(dataset, splitRatio):
    train_size = int(len(dataset) * splitRatio)
    train_set = list()
    test_set = list(dataset)
    while len(train_set) < train_size:
        index = randrange(len(test_set))
        train_set.append(test_set.pop(index))
    return [train_set, test_set]

# Function Purpose: Divide the data set by class (class is the different outcomes to train the model)
# @input: The data set with features and class in the last column
# @output: Dictionary with class as key and list of features as value
def split_dataset_by_class(dataset):
    class_and_features_dict = dict()
    for i in range(len(dataset)):
        current_row = dataset[i]
        class_value = current_row[-1]
        if (class_value not in class_and_features_dict):
            class_and_features_dict[class_value] = list()
        class_and_features_dict[class_value].append(current_row)
    return class_and_features_dict

# Function Purpose: Calculate the mean value of given class
# @input: The data set with features(input data) and class in the last column
# @output: Dictionary with class as key and list of features as value
def calculate_mean_value(features_values):
    num_of_samples = float(len(features_values))
    class_mean_value = sum(features_values) / num_of_samples 
    return class_mean_value

# Function Purpose: Calculate the stadard deviation of given class
# @input: Row of data with features(input data) of the model
# @output: stadard deviation of given class
def calculate_stdev(features_values):
    average_value = calculate_mean_value(features_values)
    num_of_samples = float(len(features_values)-1)
    variance = sum([(x-average_value)**2 for x in features_values]) / num_of_samples
    return sqrt(variance)

# Function Purpose: Calculate the mean, stdev and lenght of each feature column
# @input: The data set
# @output: A list with mean, stdev and length of each feature column
def aggregate_dataset(dataset):
    aggregated_dataset = [(calculate_mean_value(feature_column), calculate_stdev(feature_column), len(feature_column)) for feature_column in zip(*dataset)]
    del(aggregated_dataset[-1]) # exclude the class column from dataset
    return aggregated_dataset

# Function Purpose: Calculate the posteria probability: P(xi|y) = (1 / sqrt(2 * PI) * sigma) * exp(-((x-mean)^2 / (2 * sigma^2)))
# @input: feature value (single value), mean and stadard deviation
# @output: Dictionary with class as key and list of features as value
def calculate_probability(features_values, mean, stdev):
    exponent = exp(-((features_values - mean)**2 / (2 * stdev**2 )))
    posteria = (1 / (sqrt(2 * pi) * stdev)) * exponent
    return posteria

# Function Purpose: Calculate the mean, stdev and lenght of entire class
# @input: The data set
# @output: The mean, stdev and lenght of entire class
def aggregate_dataset_by_class(dataset):
    class_and_features_dict = split_dataset_by_class(dataset)
    aggregated_dataset_by_class = dict()
    for class_value, rows in class_and_features_dict.items():
        aggregated_dataset_by_class[class_value] = aggregate_dataset(rows)
    return aggregated_dataset_by_class

# Function Purpose: Calculate the class probaility:
# P(class=n|X1,X2, ..., Xn) = P(X1|class=n) * P(X2|class=n) * ... P(Xn|class=n)* P(class=n)
# @input: Aggregated data set by class and the row
# @output: The probability occurance of each class (event)
def calculate_class_probability(aggregated_dataset_by_class, row):
    class_row = 0
    dataset_len_index = 2
    total_rows = sum([aggregated_dataset_by_class[class_name][class_row][dataset_len_index] 
                      for class_name in aggregated_dataset_by_class])
    probabilities = dict()
    for class_value, class_aggregated_dataset_by_class in aggregated_dataset_by_class.items():
        probabilities[class_value] = aggregated_dataset_by_class[class_value][class_row][dataset_len_index] / (total_rows)
        for i in range(len(class_aggregated_dataset_by_class)):
            mean, stdev, count = class_aggregated_dataset_by_class[i]
            probabilities[class_value] *= calculate_probability(row[i], mean, stdev)
    return probabilities

# Function Purpose: Calculate the probability of event taken place
# @input: model data set, input data (row if data to be predicted)
# @output: The probability occurance of each class (event)
def predict_outcome(training_data, test_data):
    probabilities = calculate_class_probability(training_data, test_data)
    best_class, best_prob = None, -1
    for class_value, probability in probabilities.items():
        if best_class is None or probability > best_prob:
            best_prob = probability
            best_class = class_value
    return best_class

# Function Purpose: Collect outcomes that macthes the test data set
# @input: Data of one class, test data set
# @output: The list of matched values between the outcome and test data set
def get_predictions(summaries, testSet):
    predictions = list()
    for i in range(len(testSet)):
        result = predict_outcome(summaries, testSet[i])
        predictions.append(result)
    return predictions

# Function Purpose: Calculate the prediction accuracy
# @input: data classified by class, test training set
# @output: The prediction accuracy in percentage
def get_accuracy(test_data, prediction):
    correct = 0
    for i in range(len(test_data)):
        if test_data[i][-1] == prediction[i]: # compare prediction with the column outcome
            correct = correct + 1
            
    accuracy = (correct/float(len(test_data)))*100.0
    return accuracy

# start of the main program
if __name__ == "__main__":

    filename = 'players_stats.csv'
    dataset = load_csv(filename)
    splitRatio = 0.8 # porpotion of test dataset
    training_dataset, test_dataset = split_dataset(dataset, splitRatio)
    
    for i in range(len(dataset[0])-1):
            parse_column_to_float(dataset, i)

    parse_column_to_int(dataset, len(dataset[0])-1)
    model = aggregate_dataset_by_class(dataset)
    input_data = [50, 15, 36, 4]  #Goals, Assists, Games Played, trophies
    outcome = predict_outcome(model, input_data)
    print("---------------------------------------------------")
    print("Input Data :", input_data)
    print("Predicted Outcome: ", outcome)
    print("---------------------------------------------------")
    predictions = get_predictions(model, test_dataset)
    accuracy = get_accuracy(test_dataset, predictions)
    print("Prediction accuracy :", accuracy, "%")
