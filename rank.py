import os
import pickle
from keras.models import load_model
from keras.datasets import imdb
from keras.preprocessing import sequence

# load dataset
Dataset = imdb.load_data(num_words=10000)
X_val = sequence.pad_sequences(Dataset[1][0], maxlen=200)
Y_val = Dataset[1][1]

folder = "savedModels" 

for filename in os.listdir(folder):
    if filename.endswith(".keras"):
        path = os.path.join(folder, filename)
        model = load_model(path)
        loss, accuracy = model.evaluate(X_val, Y_val, verbose=0)  # verbose=0 suppresses progress bar
        print(f"{filename}: val_accuracy = {accuracy:.4f}")