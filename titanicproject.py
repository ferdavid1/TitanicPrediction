# Import libraries necessary for this project
import numpy as np
import pandas as pd
from IPython.display import display # Allows the use of display() for DataFrames

# Import supplementary visualizations code visuals.py
import visuals as vs

# Pretty display for notebooks
#%matplotlib inline

# Load the dataset
in_file = 'titanic_data.csv'
full_data = pd.read_csv(in_file)

# Print the first few entries of the RMS Titanic data
###display(full_data.head())
# Store the 'Survived' feature in a new variable and remove it from the dataset, because that is what we are going to predict if they will survive or not
outcomes = full_data['Survived']
data = full_data.drop('Survived', axis = 1)

# Show the new dataset with 'Survived' removed
display(data.head())
def accuracy_score(truth, pred):
    """ Returns accuracy score for input truth and predictions. """
    
    # Ensure that the number of predictions matches number of outcomes
    if len(truth) == len(pred): 
        
        # Calculate and return the accuracy as a percent
        return "Predictions have an accuracy of {:.2f}%.".format((truth == pred).mean()*100)
    
    else:
        return "Number of predictions does not match number of outcomes!"
    
# Test the 'accuracy_score' function
predictions = pd.Series(np.ones(5, dtype = int))
print (accuracy_score(outcomes[:5], predictions))
def predictions_0(data):
    """ Model with no features. Always predicts a passenger did not survive. """

    predictions = []
    for _, passenger in data.iterrows():
        
        # Predict the survival of 'passenger'
        predictions.append(0)
    
    # Return our predictions
    return pd.Series(predictions)

# Make the predictions
predictions = predictions_0(data)
print(accuracy_score(outcomes, predictions)) #compare prediction where no one survives to prediction accuracy score
###vs.survival_stats(data, outcomes, 'Sex')
def predictions_1(data):
    """ Model with one feature: 
            - Predict a passenger survived if they are female. """
    
    predictions = []
    for _, passenger in data.iterrows():
        if passenger['Sex'] == 'female':
        	predictions.append(1)
        else:
        	predictions.append(0)
    # Return our predictions
    return pd.Series(predictions)

# Make the predictions
predictions = predictions_1(data)
print(accuracy_score(outcomes, predictions))
###vs.survival_stats(data, outcomes, 'Age', ["Sex == 'male'"])
def predictions_2(data):
    """ Model with two features: 
            - Predict a passenger survived if they are female.
            - Predict a passenger survived if they are male and younger than 10. """
    
    predictions = []
    for _, passenger in data.iterrows():
    	if passenger['Sex'] == 'female':
    		predictions.append(1)
    	elif passenger['Sex'] == 'male' and passenger['Age']<10:
    		predictions.append(1)
    	else:
    		predictions.append(0)
    # Return our predictions
    return pd.Series(predictions)

# Make the predictions
predictions = predictions_2(data)
print(accuracy_score(outcomes, predictions))
vs.survival_stats(data, outcomes, 'Fare', ["Sex == 'female'"])
def predictions_3(data):
    """ Model with three features: """
    
    predictions = []
    for _, passenger in data.iterrows(): 
    	if passenger['Sex'] == 'female':
    		if passenger['SibSp']<=2:
    			predictions.append(1)
    		else:
    			predictions.append(0)
    	elif passenger['Sex'] == 'male' and passenger['Age']<10:
    		predictions.append(1)
    	elif passenger['Fare']>263:
    		predictions.append(1)
    	else:
    		predictions.append(0)
    # Return our predictions
    return pd.Series(predictions)

# Make the predictions
predictions = predictions_3(data)
print(accuracy_score(outcomes, predictions))