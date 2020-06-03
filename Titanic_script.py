import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#Loading the passenger data
passengers = pd.read_csv('passengers.csv')


#Updating sex column to numerical
passengers['Sex'] = passengers['Sex'].map({
  'male' : 0,
  'female': 1
})


# Filling the nan values in the age column

passengers['Age'].fillna(value = np.mean(passengers['Age']), inplace = True)

# Creating a first class column
passengers['FirstClass'] = passengers['Pclass'].apply(lambda x: 1 if x == 1 else 0)



# Creating a second class column
passengers['SecondClass'] = passengers['Pclass'].apply(lambda x: 1 if x == 2 else 0)

# Selecting the desired features
features = passengers[['Sex', 'Age', 'FirstClass', 'SecondClass']]
survival = passengers['Survived']
# Perform train, test, split
train_features, test_features, train_labels, test_labels = train_test_split(features, survival, test_size = 0.2)



# Scaling the feature data

scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
test_features = scaler.transform(test_features)

# Creating and training the model
model = LogisticRegression()
model.fit(train_features, train_labels)

# Scoring the model on the train data
print(model.score(train_features, train_labels))

# Scoring the model on the test data
print(model.score(test_features, test_labels))
print(model.coef_)
# Analyzing the coefficients
print(list(zip(['Sex','Age','FirstClass','SecondClass'],model.coef_[0])))

# Sample passenger features
Jack = np.array([0.0,20.0,0.0,0.0])
Rose = np.array([1.0,17.0,1.0,0.0])
You = np.array([0.0,20.0,1.0,0.0])

# Combining passenger arrays
combined_array = np.array([Jack, Rose, You])
print(combined_array)
# Scaling the sample passenger features
sample_passengers = scaler.transform(combined_array)
print(sample_passengers)

# Making survival predictions!
print(model.predict(sample_passengers))
print(model.predict_proba(sample_passengers))
