import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.nn.functional as F
import tensorflow as tf
from tensorflow.keras import layers, losses, optimizers, callbacks

#try to only feed the ROI selections into the autoencoder w/ modification
#just extract data that is other than background or tissues. LImiting it to only ROI (Helps reduce the size)
#use a dr approach to reduce dimensionality to a resonable point (we need a higher one for classification)
#Change things to a dimension reduct and then classification. What happens if we use classical methods (PCA + LDA) as comapred to using deep learning approach
# Compare the results between the two. What is the difference between the two? PCA LDA vs machine learning DR and classification
# Add tissue data as well into the DR (will this imrpove the classification accuracy). This means we used unlabelled data to improve accuracy
# for the AE, we have another dense layer (recieves teh DR reduced data) and classify the data. All labeled data.
#ask GPT to create an AE with dataload with validation. Then use the AE to classify the data. Then compare the results
#read csv, and only keep the data of the 6 labels, then recreate the csv file and open on slicer
#you can fake the m/z it does not matter for the slicer

# Define the dataset
class MassSpecDataset(Dataset):
    def __init__(self, data, labels=None):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.data[idx], self.labels[idx]
        else:
            return self.data[idx]

class Autoencoder(tf.keras.Model):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(input_dim,)),
            layers.Dense(latent_dim, activation='relu')
        ])

        self.decoder = tf.keras.Sequential([
            layers.Dense(128, activation='relu'),
            layers.Dense(input_dim, activation='sigmoid')
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class CNNClassifier(tf.keras.Model):
    def __init__(self, input_dim, dropout_rate=0.3, momentum=0.1):
        super(CNNClassifier, self).__init__()
        self.fc1 = layers.Dense(1024, activation='relu', input_shape=(input_dim,))
        self.dropout1 = layers.Dropout(dropout_rate)
        self.bn1 = layers.BatchNormalization(momentum=momentum)

        self.fc2 = layers.Dense(512, activation='relu')
        self.dropout2 = layers.Dropout(dropout_rate)
        self.bn2 = layers.BatchNormalization(momentum=momentum)

        self.fc3 = layers.Dense(256, activation='relu')
        self.dropout3 = layers.Dropout(dropout_rate)
        self.bn3 = layers.BatchNormalization(momentum=momentum)

        self.fc4 = layers.Dense(128, activation='relu')
        self.dropout4 = layers.Dropout(dropout_rate)
        self.bn4 = layers.BatchNormalization(momentum=momentum)

        self.fc5 = layers.Dense(64, activation='relu')
        self.dropout5 = layers.Dropout(dropout_rate)
        self.bn5 = layers.BatchNormalization(momentum=momentum)

        self.fc6 = layers.Dense(6, activation='softmax')

    def call(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = self.dropout3(x)

        x = self.fc4(x)
        x = self.bn4(x)
        x = self.dropout4(x)

        x = self.fc5(x)
        x = self.bn5(x)
        x = self.dropout5(x)

        x = self.fc6(x)
        return x
