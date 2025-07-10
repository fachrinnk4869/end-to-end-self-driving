import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load loss data from file
# Assuming the file contains two columns: train_loss and val_loss
data = pd.read_csv('trainval_log.csv', delimiter=',')
train_loss = data['train_loss'].values
val_loss = data['val_loss'].values

# Plot train and validation loss
plt.figure(figsize=(10, 6))
plt.plot(train_loss, label='Train Loss', color='blue')
plt.plot(val_loss, label='Validation Loss', color='orange')

# Highlight the decreasing validation loss
min_val_loss = val_loss[0]
for i in range(1, len(val_loss)):
    if val_loss[i] < min_val_loss:
        min_val_loss = val_loss[i]
        plt.hlines(y=val_loss[i], xmin=i-1, xmax=i+1,
                   color='green', linestyle='-', alpha=0.5)

# Show all x-axis values
epochs = np.arange(len(val_loss))
plt.xticks(epochs)
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(False)
plt.show()
