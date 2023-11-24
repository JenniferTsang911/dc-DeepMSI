from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch

def get_test_train(output, csv_path, patch_size):
    # Extract patches from the output feature maps
    output_patches = output.unfold(1, patch_size, 1).unfold(2, patch_size, 1)
    print(output_patches.size())

    # Flatten the patches
    output_patches_flattened = output_patches.reshape(output_patches.shape[0], -1)
    print(output_patches_flattened.shape)

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

                features.append(np.concatenate([output_patches_flattened[y * output.shape[1] + x, :], df.loc[
                                                    (df['X'] == x) & (df['Y'] == y), df.columns[3:]].mean().values]))

                # Add the most common class label to the labels
                labels.append(most_common_label)

    # print(features)
    # print(labels)

    # Step 1: Handle NaN values
    features = np.nan_to_num(features, nan=0.0)

    # Step 2: Handle infinity values
    features = np.where(np.isinf(features), np.finfo('float64').max, features)


    # Encode the class labels as integers
    le = LabelEncoder()
    labels = le.fit_transform(labels)

    # Split the features and labels into a training set and a test set
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2,
                                                                                random_state=42)

    return features_train, features_test, labels_train, labels_test
def logisticRegression(features_train, features_test, labels_train, labels_test):
    # Train the logistic regression model

    scaler = StandardScaler()
    features_train = scaler.fit_transform(features_train)
    features_test = scaler.transform(features_test)

    print("Training")
    clf = LogisticRegression(max_iter=1000, solver= 'newton-cg')
    clf.fit(features_train, labels_train)


    print("Predicting")
    predicted_labels = clf.predict(features_test)

    accuracy = accuracy_score(labels_test, predicted_labels)
    print("Accuracy: " + str(accuracy))
    c_matrix = confusion_matrix(labels_test, predicted_labels)
    print("Confusion Matrix: " + str(c_matrix))

    # # visualize the predicted labels
    # # Assume that `clf` is your trained classifier and `features` is your input data
    # predicted_labels = clf.predict(features)
    #
    # # Get the shape of the original output
    # original_shape = output.shape
    #
    # # Make sure the total size of the new shape is equal to the size of the original array
    # assert np.prod(original_shape) == len(predicted_labels)
    #
    # # Reshape the predicted labels to match the shape of the original output
    # predicted_labels_reshaped = predicted_labels.reshape(original_shape)
    #
    # # Display the reshaped labels as an image
    # plt.imshow(predicted_labels_reshaped, cmap='tab20')
    # plt.show()

    return clf



def runLDA(features_train, features_test, labels_train, labels_test):
    pca_model = PCA(n_components=0.99)
    pca_model.fit(features_train)
    print('number of PC:', pca_model.n_components_)
    features_train_pca = pca_model.transform(features_train)
    features_test_pca = pca_model.transform(features_test)

    n_class = len(np.unique(labels_train))
    lda_model = LDA(n_components=n_class - 1)
    lda_model.fit(features_train_pca, labels_train)
    features_train_lda = lda_model.transform(features_train_pca)
    labels_train_preds = lda_model.predict(features_train_pca)
    labels_train_prob = lda_model.predict_proba(features_train_pca)
    features_test_lda = lda_model.transform(features_test_pca)
    labels_test_preds = lda_model.predict(features_test_pca)
    labels_test_prob = lda_model.predict_proba(features_test_pca)

    class_order = lda_model.classes_

    scatter_data = features_train_lda[:, :2]
    scatter_labels = labels_train

    scatter_normalizaer = MinMaxScaler()
    scatter_data = scatter_normalizaer.fit_transform(scatter_data)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    legend_labels = np.unique(scatter_labels)
    n_colors = len(legend_labels)
    if n_colors <= 10:
        class_colors = plt.cm.tab10(range(n_colors))
    else:
        class_colors = cm.get_cmap('jet_r')(np.linspace(0, 1, n_colors))

    for i in range(len(legend_labels)):
        ind = scatter_labels == legend_labels[i]
        xx = scatter_data[ind]
        ax.scatter(xx[:, 0], xx[:, 1],
                   color=class_colors[i],
                   label=legend_labels[i], alpha=0.8)

    plt.legend()
    ax.set_xlabel('LDA1')
    ax.set_ylabel('LDA2')
    ax.set_ylim([0, 1])
    ax.set_xlim([0, 1])

    # save plot
    filename = 'LDAplot.jpeg'
    plt.savefig(filename, bbox_inches='tight', dpi=600)
    plt.close()

    return labels_train_preds, labels_train_prob, labels_test_preds, labels_test_prob, class_order, [pca_model, lda_model]



csv_path = r"C:\Users\tsang\Documents\GitHub\DESI-project\2021 03 29 colon 0462641-2 Analyte 2_dataset balanced.csv"
patch_size = 5

output = np.load('outputAverage.npy')

output = torch.from_numpy(output)
#torch.Size([51980, 30]) torch.Size([51980, 30])

features_train, features_test, labels_train, labels_test = get_test_train(output, csv_path, patch_size)

#runLDA(features_train, features_test, labels_train, labels_test)
logisticRegression(features_train, features_test, labels_train, labels_test)