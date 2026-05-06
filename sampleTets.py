

import matplotlib
matplotlib.use("TkAgg")
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from tensorflow import keras
import pickle


Dataset = imdb.load_data(num_words=10000)
# Zero-Padding

maxLength = 300 # hyperparam





X_train = sequence.pad_sequences(Dataset[0][0], maxlen=maxLength)
Y_train = Dataset[0][1]
X_val = sequence.pad_sequences(Dataset[1][0], maxlen=maxLength)
Y_val = Dataset[1][1]


# model name to load

modelName = "initial"
modelName = "rmsprop-oDim-32-bSize-32-e-12-mLen-300"

# load the model
model = keras.models.load_model("savedModels/"+modelName+".keras")

#with open("savedModels/"+modelName+'.pkl', 'rb') as f:
#    history = pickle.load(f)

# print accuracy of model
#(loss, accuracy) = model.evaluate(X_val, Y_val)
#print("Test Accuracy: ",accuracy)


test = 35

# grab one sample from validation set
sample = X_val[test].reshape(1, 300)  # reshape to batch of 1
probability = model.predict(sample)[0][0]
sentiment = "Positive" if probability > 0.5 else "Negative"
print(f"Probability: {probability:.4f}")
print(f"Sentiment: {sentiment}")

# check what the actual label was
print(f"Actual: {'Positive' if Y_val[test] == 1 else 'Negative'}")




# Plot accuracy per epoch

#print("Printing accuracy graphs")
#plt.plot(history["accuracy"], label="Accuracy")
#plt.plot(history["val_accuracy"], label="Validation Accuracy") # validation accuracy is more useful here
#plt.title("Accuracy per Epoch")
#plt.xlabel("Epoch")
#plt.ylabel("Accuracy")
#plt.legend()
#plt.show()


# Plot confusion matrix

#y_pred = (model.predict(X_val) > 0.5).astype(int)
#print(y_pred)
#y_pred=y_pred.flatten()
#print(y_pred)
#
#confusionMatrix = confusion_matrix(Y_val, y_pred)
## code referenced from medium.com blog to help create useful matrix.
#sns.heatmap(confusionMatrix, annot=True, fmt='d', cmap='Blues',
#            xticklabels=['Negative', 'Positive'],
#            yticklabels=['Negative', 'Positive'])
#plt.title('Confusion Matrix')
#plt.xlabel('Predicted')
#plt.ylabel('Actual')
#plt.show()
#
#
#write about lstm and shits for model
