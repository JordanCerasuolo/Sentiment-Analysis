

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

# Import IMDB dataset keeping only 10,000 most common unique words
Dataset = imdb.load_data(num_words=10000)
# Zero-Padding

maxLength = 300 # hyperparam

X_train = sequence.pad_sequences(Dataset[0][0], maxlen=maxLength)
Y_train = Dataset[0][1]
X_val = sequence.pad_sequences(Dataset[1][0], maxlen=maxLength)
Y_val = Dataset[1][1]
# Model Building

output_dim=32 # hyperparam
batch_size=32 # hyperparam
epochs=12 # hyperparam

def train_model(Optimizer, X_train, Y_train, X_val, Y_val):
    model = Sequential() # MODEL USED!
    # **ADD YOUR CODE HERE**

    # add layers
    
    model.add(Embedding(input_dim=10000, output_dim=output_dim)) # 10,000 due to only 10,000 common words
    model.add(LSTM(units=50, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)) # add dropout to reduce overfitting
    model.add(LSTM(units=50, dropout=0.2, recurrent_dropout=0.2)) # add dropout to reduce overfitting, layer 2
    
    model.add(Dense(units=1, activation="sigmoid")) # 1 unit because binary classification
    
    model.compile(
        optimizer=Optimizer,
        loss="binary_crossentropy", # best for binary classification, no change
        metrics=["accuracy"],
    )

    fitted = model.fit(x=X_train, y=Y_train,batch_size=batch_size,epochs=epochs,validation_data=(X_val,Y_val))



    return (fitted, model)

# Train Model

optimizer = "rmsprop" # hyperparam



trained = train_model(Optimizer=optimizer, X_train=X_train,Y_train=Y_train,X_val=X_val,Y_val=Y_val)

fittedModel, model = trained

with open("savedModels/"+str(optimizer)+"-oDim-"+str(output_dim)+"-bSize-"+str(batch_size)+"-e-"+str(epochs)+"-mLen-"+str(maxLength)+'.pkl', 'wb') as f:
    pickle.dump(fittedModel.history, f)
model.save("savedModels/"+str(optimizer)+"-oDim-"+str(output_dim)+"-bSize-"+str(batch_size)+"-e-"+str(epochs)+"-mLen-"+str(maxLength)+'.keras')

# print accuracy of model
(loss, accuracy) = model.evaluate(X_val, Y_val)
print("Test Accuracy: ",accuracy)

# Plot accuracy per epoch

print("Printing accuracy graphs")
plt.plot(fittedModel.history["accuracy"], label="Accuracy")
plt.plot(fittedModel.history["val_accuracy"], label="Validation Accuracy") # validation accuracy is more useful here
plt.title("Accuracy per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


# Plot confusion matrix

y_pred = (model.predict(X_val) > 0.5).astype(int)
print(y_pred)
y_pred=y_pred.flatten()
print(y_pred)

confusionMatrix = confusion_matrix(Y_val, y_pred)
# code referenced from medium.com blog to help create useful matrix.
sns.heatmap(confusionMatrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


#write about lstm and shits for model
