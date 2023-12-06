from sklearn.cluster import KMeans
import torch
import numpy as np
from Model import *
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

data = np.loadtxt(r"C:\Users\jenni\OneDrive - Queen's University\DESI project\DESI TXT colon\Arrays\2021 03 30 colon 0413337-2 Analyte 6 array.txt")

m, n = data.shape

# Load the state dictionary from the saved model file
pretrain = torch.load('FEmodule_weight - Copy.pth')

# Initialize the Autoencoder model
autoencoder = Autoencoder(n)

# Get the state dictionary of the initialized model
model_dict = autoencoder.state_dict()

# Filter out the mismatched weights from the loaded state dictionary
pretrain = {k: v for k, v in pretrain.items() if
            k in model_dict.keys() and v.shape == model_dict[k].shape}

# Update the state dictionary of the initialized model with the filtered state dictionary
model_dict.update(pretrain)

# Load the updated state dictionary into the model
autoencoder.load_state_dict(model_dict)
autoencoder.eval()

# Convert your data to a PyTorch tensor
data_tensor = torch.from_numpy(data.astype(np.float32))

# Pass your data through the encoder
with torch.no_grad():
    encoded_data, _ = autoencoder(data_tensor)

# Convert the encoded data to a numpy array
encoded_data_np = encoded_data.numpy()

#encoded_data_np = np.load(r"C:\Users\jenni\Documents\GitHub\dc-DeepMSI\encoded_output.npy")

# Perform k-means clustering on the encoded data
kmeans = KMeans(n_clusters=12)
kmeans.fit(encoded_data_np)

# Reshape the labels to the original shape of your data
segmentation_map = kmeans.labels_.reshape(205, 263)

# Plot the segmentation map
plt.imshow(segmentation_map, cmap='viridis')
plt.colorbar()
plt.title('Segmentation Map')
plt.show()