from AE_Classifier_Model import *
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import matplotlib.pyplot as plt
from Loss import CosineLoss
from sklearn.model_selection import GridSearchCV

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, optimizers

def process_data(input_file):
    # Load the CSV file
    df = pd.read_csv(input_file)
    df = df.loc[df['Class'] != 'tissue']

    # Encode the labels
    le = LabelEncoder()
    df['Class'] = le.fit_transform(df['Class'])

    # Get a list of unique slide names
    slide_names = df['Slide'].unique()

    # Split the slide names into training and temporary sets
    train_slide_names, temp_slide_names = train_test_split(slide_names, test_size=0.5, random_state=42)

    # Split the temporary set into validation and test sets
    val_slide_names, test_slide_names = train_test_split(temp_slide_names, test_size=0.6, random_state=42)

    # Create the training, validation, and test DataFrames
    train_df = df[df['Slide'].isin(train_slide_names)]
    val_df = df[df['Slide'].isin(val_slide_names)]
    test_df = df[df['Slide'].isin(test_slide_names)]

    print("Training slides:", train_df['Slide'].unique())
    print("Validation slides:", val_df['Slide'].unique())
    print("Test slides:", test_df['Slide'].unique())

    # Reset the index of the DataFrames
    train_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    # Separate the features and labels in the training, validation, and test dataframes
    train_features = train_df[df.columns[4:]]
    train_labels = train_df['Class']
    val_features = val_df[df.columns[4:]]
    val_labels = val_df['Class']
    test_features = test_df[df.columns[4:]]
    test_labels = test_df['Class']

    # Convert the features to TensorFlow tensors
    train_data = tf.convert_to_tensor(train_features.values, dtype=tf.float32)
    val_data = tf.convert_to_tensor(val_features.values, dtype=tf.float32)
    test_data = tf.convert_to_tensor(test_features.values, dtype=tf.float32)

    # Convert the labels to TensorFlow tensors
    train_labels = tf.convert_to_tensor(train_labels.values, dtype=tf.int32)
    val_labels = tf.convert_to_tensor(val_labels.values, dtype=tf.int32)
    test_labels = tf.convert_to_tensor(test_labels.values, dtype=tf.int32)

    return train_data, train_labels, val_data, val_labels, test_data, test_labels

def train_autoencoder(train_data, val_data, latent_dim=100, num_epochs=50, learning_rate=0.001):
    input_dim = train_data.shape[1]
    autoencoder = Autoencoder(input_dim, latent_dim)

    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss='mse')

    history = autoencoder.fit(train_data, train_data,
                              epochs=num_epochs,
                              validation_data=(val_data, val_data))

    # Use the encoder to perform dimensionality reduction
    train_data_encoded = autoencoder.encoder(train_data).numpy()
    val_data_encoded = autoencoder.encoder(val_data).numpy()

    return autoencoder, train_data_encoded, val_data_encoded, history

def train_classifier(train_data, train_labels, val_data, val_labels, num_epochs_CL=200, CL_LR=0.001, dropoutCL=0.5, momentumCL=0.1, l2_CL=0.01):
    print(train_data.shape)

    train_loader = DataLoader(MassSpecDataset(train_data, train_labels), batch_size=64, shuffle=True)
    val_loader = DataLoader(MassSpecDataset(val_data, val_labels), batch_size=64)

    print('Training classifier...')
    model = CNNClassifier(train_data.shape[1], dropoutCL, momentumCL).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CL_LR, weight_decay=l2_CL)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    loss_values_classifier = []
    val_loss_values_classifier = []
    best_val_loss_CL = float('inf')
    epochs_no_improve_CL = 0
    patience = 50

    for epoch in range(num_epochs_CL):
        for batch_data, batch_labels in train_loader:
            batch_labels = batch_labels.clone().detach()
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward(retain_graph=True)
            optimizer.step()

        scheduler.step()

        val_loss_CL = 0
        with torch.no_grad():
            for val_data, val_labels in val_loader:
                val_outputs = model(val_data)
                val_loss_CL += criterion(val_outputs, val_labels).item()
        val_loss_CL /= len(val_loader)
        val_loss_values_classifier.append(val_loss_CL)

        if val_loss_CL < best_val_loss_CL:
            best_val_loss_CL = val_loss_CL
            epochs_no_improve_CL = 0
        else:
            epochs_no_improve_CL += 1

        if epochs_no_improve_CL == patience:
            print('Early stopping!')
            break

        loss_values_classifier.append(loss.item())
        print(f"Classifier Epoch {epoch+1}/{num_epochs_CL} - Loss: {loss.item()} - Val Loss: {val_loss_CL}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return model, loss_values_classifier, val_loss_values_classifier

def evaluate_model(autoencoder, model, test_loader):
    model.eval()  # Set the model to evaluation mode
    all_labels = []
    all_predictions = []

    with torch.no_grad():  # No need to calculate gradients
        for data, labels in test_loader:
            encoded_data, _ = autoencoder(data)  # Pass the test data through the autoencoder
            outputs = model(encoded_data)  # Use the encoded test data
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu())  # Move labels to CPU
            all_predictions.extend(predicted.cpu())  # Move predicted to CPU

    # Convert to numpy arrays for use with sklearn metrics
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)

    # Display confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)

    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')  # Save the plot to a file

    print("Precision:", precision_score(all_labels, all_predictions, average='weighted'))
    print("Recall:", recall_score(all_labels, all_predictions, average='weighted'))
    print("F1 Score:", f1_score(all_labels, all_predictions, average='weighted'))

def main():
    num_epochs_AE = 1000
    AE_LR = 0.0001
    l2_AE = 0.01

    num_epochs_CL = 500
    CL_LR = 0.0001
    dropoutCL = 0.5
    momentumCL = 0.99
    l2_CL = 0.0001

    train_data, train_labels, val_data, val_labels, test_data, test_labels = process_data(r"C:\Users\jenni\Documents\GitHub\dc-DeepMSI\all_aligned_no_background_others_preprocessed.csv")

    #autoencoder, train_data, val_data, loss_values_autoencoder, val_loss_values_autoencoder = Dimension_Reduction(train_data, train_labels, val_data, num_epochs_AE, AE_LR)

    autoencoder, train_data_encoded, val_data_encoded, history = train_autoencoder(train_data, val_data)

    #model, loss_values_classifier, val_loss_values_classifier = train_classifier(train_data, train_labels, val_data, val_labels, num_epochs_CL, CL_LR, dropoutCL, momentumCL, l2_CL)

    # Create a DataLoader for the test data
    #test_loader = DataLoader(MassSpecDataset(test_data, test_labels), batch_size=64)

    #evaluate_model(autoencoder,model, test_loader)

    # Create a figure and a set of subplots
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))  # Adjust the figsize as needed

    # Plot the training and validation loss curves for the autoencoder
    axs[0].plot(history.history['loss'], label='Training Loss - Autoencoder')
    axs[0].plot(history.history['val_loss'], label='Validation Loss - Autoencoder')
    axs[0].set_title('Loss curves for the Autoencoder')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend()

    # Plot the training and validation loss curves for the classifier
    axs[1].plot(loss_values_classifier, label='Training Loss - Classifier')
    axs[1].plot(val_loss_values_classifier, label='Validation Loss - Classifier')
    axs[1].set_title('Loss curves for the Classifier')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Loss')
    axs[1].legend()

    # Display the figure
    plt.tight_layout()
    plt.show()

main()
'''
def process_data(input_file):
    # Load the CSV file
    df = pd.read_csv(input_file)

    # Encode the labels
    le = LabelEncoder()
    df['Class'] = le.fit_transform(df['Class'])

    # Get the encoded value for 'tissues'
    tissues_class_value = le.transform(['tissues'])[0]

    # Rest of your code...

    return train_data, train_labels, val_data, val_labels, test_data, test_labels, tissues_class_value

def train_model(train_data, train_labels, val_data, val_labels, test_data, test_labels, tissues_class_value, num_epochs_AE=1000, num_epochs_CL=200, AE_LR=0.001, CL_LR=0.001, dropoutCL=0.5, momentumCL=0.1, momentumAE=0.1):
    # Rest of your code...

    # Train the autoencoder
    for epoch in range(num_epochs_AE):
        # Your training code...

    # Remove the 'tissues' class from the training data and the labels
    tissues_class_index = (train_labels != tissues_class_value)
    train_data = train_data[tissues_class_index]
    train_labels = train_labels[tissues_class_index]

    # Rest of your code...

    return model, loss_values_classifier, loss_values_autoencoder, val_loss_values_classifier, val_loss_values_autoencoder, test_data'''