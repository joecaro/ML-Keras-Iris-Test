import pandas as pd
from keras.models import Sequential
from keras.layers import *
from sklearn.model_selection import train_test_split

# Import Data
data_df = pd.read_csv('../Process Data/scaled_encoded_data.csv')


# Split Data into Training and Testing Samples
data_train, data_test = train_test_split( data_df, test_size=.33, random_state=42)


# Remake the Training DataFrame & Separate Inputs and Outputs
training_data_df = pd.DataFrame(data_train, columns=data_df.columns.values)
X = training_data_df.drop(['setosa', 'versicolor', 'virginica'], axis=1).values
Y = training_data_df[['setosa', 'versicolor', 'virginica']].values


# TRAINING

# Define the Model
model = Sequential()
model.add(Dense(10, input_dim=4, activation='relu'))
model.add(Dense(3, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='accuracy')

# Train the Model
model.fit(
    X,
    Y,
    epochs=500,
    shuffle=True,
    verbose=2
)

# TESTING

# Remake the Testing DataFrame & Separate Inputs and Outputs
testing_data_df = pd.DataFrame(data_test, columns=data_df.columns.values)
X_test = testing_data_df.drop(['setosa', 'versicolor', 'virginica'], axis=1).values
Y_test = testing_data_df[['setosa', 'versicolor', 'virginica']].values

# Test the data
test_error_rate = model.evaluate(X_test, Y_test, verbose=0)
print("The mse for the test dataset is: {}".format(test_error_rate))
