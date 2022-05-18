from numpy import array
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

# define example
data = ['cold', 'cold', 'warm', 'cold', 'hot', 'hot', 'warm', 'cold', 'warm', 'hot']
values = array(data)
print(values)
# integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)
print(integer_encoded)
# one hot encode
encoded = to_categorical(integer_encoded)
print(encoded)