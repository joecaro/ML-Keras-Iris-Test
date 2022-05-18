import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# Load dataset
data_df = pd.read_csv('iris.csv')

# Separate Inputs
inputs_df = data_df.drop(columns='class')

# Separate Outputs
outputs_df = data_df[['class']].values
# Binary Encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(outputs_df)

onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

outputs = onehot_encoded


# Scale Input data to fit between 0 adn 1
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_inputs = scaler.fit_transform(inputs_df)

# Create new DataFrame objects from scaled data
scaled_inputs_df = pd.DataFrame(scaled_inputs, columns=inputs_df.columns.values)
encoded_outputs_df = pd.DataFrame(outputs, columns=['setosa', 'versicolor', 'virginica'])

# Concat Data
scaled_encoded_data_df = pd.concat([scaled_inputs_df, encoded_outputs_df], axis=1, sort=False)

# Save scaled/encoded data to new CSV files
# scaled_inputs_df.to_csv('scaled_input_data.csv', index=False)
# encoded_outputs_df.to_csv('encoded_outputs.csv', index=False)
# scaled_encoded_data_df.to_csv('scaled_encoded_data.csv', index=False)
# print('New CSVs Saved')