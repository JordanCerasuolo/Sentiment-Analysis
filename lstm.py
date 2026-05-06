

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

# Import IMDB dataset keeping only 10,000 most common unique words
Dataset = imdb.load_data(num_words=10000)
# Zero-Padding

X_train = sequence.pad_sequences(Dataset[0][0], maxlen=100)
Y_train = Dataset[0][1]
X_val = sequence.pad_sequences(Dataset[1][0], maxlen=100)
Y_val = Dataset[1][1]
# Model Building

output_dim=50
batch_size=20
epochs=5

def train_model(Optimizer, X_train, Y_train, X_val, Y_val):
    model = Sequential()
    # **ADD YOUR CODE HERE**

    # add layers
    
    model.add(Embedding(input_dim=10000, output_dim=output_dim)) # 10,000 due to only 10,000 common words
    model.add(LSTM(units=50))
    model.add(Dense(units=1, activation="sigmoid"))
    
    model.compile(
        optimizer=Optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    fitted = model.fit(x=X_train, y=Y_train,batch_size=batch_size,epochs=epochs,validation_data=(X_val,Y_val))



    return (fitted, model)

# Train Model

trained = train_model(Optimizer="rmsprop", X_train=X_train,Y_train=Y_train,X_val=X_val,Y_val=Y_val)

fittedModel, model = trained

# print accuracy of model
(loss, accuracy) = model.evaluate(X_val, Y_val)
print("Test Accuracy: ",accuracy)

# Plot accuracy per epoch

print("Printing accuracy graphs")
plt.plot(fittedModel.history['accuracy'], label="Accuracy")
plt.plot(fittedModel.history['val_accuracy'], label="Validation Accuracy")
plt.title('Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
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


