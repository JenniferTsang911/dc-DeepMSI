from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
def logisticRegression(output, output2, csv_path, patch_size):
    # Extract patches from the output feature maps
    output_patches = output.unfold(1, patch_size, 1).unfold(2, patch_size, 1)
    output2_patches = output2.unfold(1, patch_size, 1).unfold(2, patch_size, 1)
    print(output_patches.size())

    # Flatten the patches
    output_patches_flattened = output_patches.reshape(output_patches.shape[0], -1)
    output2_patches_flattened = output2_patches.reshape(output2_patches.shape[0], -1)
    print(output_patches_flattened.shape)
    print(output2_patches_flattened.shape)

    # Load the CSV file
    df = pd.read_csv(csv_path)
    df.fillna(0, inplace=True)

    # Create a dictionary where the keys are the (X, Y) coordinates and the values are the class labels
    csv_dict = {(row['X'], row['Y']): row for _, row in df.iterrows()}

    # Initialize the features and labels
    features = []
    labels = []


    # For each patch
    for y in range(0, output.shape[0] - patch_size + 1):
        for x in range(0, output.shape[1] - patch_size + 1):
            # Get the class labels for the corresponding region in the CSV file
            patch_labels = [csv_dict.get((x + dx, y + dy))['Class'] for dx in range(patch_size) for dy in range(patch_size) if csv_dict.get((x + dx, y + dy)) is not None]

            # If there are any class labels for the corresponding region in the CSV file
            if patch_labels:
                # Get the most common class label in the region
                most_common_label = stats.mode(patch_labels)[0][0]

                # Add the feature vector for the pixel, the corresponding m/z values, and the class label to the features
                #features.append(np.concatenate([output_patches_flattened[y * output.shape[1] + x, :], output2_patches_flattened[y * output.shape[1] + x, :], df.loc[(df['X'] == x) & (df['Y'] == y), df.columns[3:]].values]))

                features.append(np.concatenate([output_patches_flattened[y * output.shape[1] + x, :],
                                                output2_patches_flattened[y * output.shape[1] + x, :], df.loc[
                                                    (df['X'] == x) & (df['Y'] == y), df.columns[3:]].mean().values]))

                # Add the most common class label to the labels
                labels.append(most_common_label)

    # print(features)
    # print(labels)


    # Encode the class labels as integers
    le = LabelEncoder()
    labels = le.fit_transform(labels)

    # Split the features and labels into a training set and a test set
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2,
                                                                                random_state=42)

    print(len(features_train))
    print(len(labels_train))

    # Train the logistic regression model
    print("Training")
    clf = LogisticRegression()
    clf.fit(features_train, labels_train)


    print("Predicting")
    predicted_labels = clf.predict(features_test)

    accuracy = accuracy_score(labels_test, predicted_labels)
    print("Accuracy: " + str(accuracy))
    c_matrix = confusion_matrix(labels_test, predicted_labels)
    print("Confusion Matrix: " + str(c_matrix))

    return clf

#visualize the predicted labels
# # Assume that `clf` is your trained classifier and `features` is your input data
# predicted_labels = clf.predict(features)
#
# # Reshape the predicted labels to match the shape of the original image
# predicted_labels_reshaped = predicted_labels.reshape(output.shape[0], output.shape[1])
#
# # Display the reshaped labels as an image
# plt.imshow(predicted_labels_reshaped, cmap='tab20')
# plt.show()

csv_path = r"balanced_dataset.csv"
patch_size = 5

output = np.load('output.npy')
output2 = np.load('output2.npy')

output = torch.from_numpy(output)
output2 = torch.from_numpy(output2)
#torch.Size([51980, 30]) torch.Size([51980, 30])

logisticRegression(output, output2, csv_path, patch_size)