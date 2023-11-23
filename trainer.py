import torch.nn.init
import csv
from sklearn.feature_extraction import image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy import stats
import pandas as pd
from Loss import *
from Model import *

def Dimension_Reduction(data_image, args, pretrain_path=None):
    print('Step 1: Dimensionality Reduction ...')
    m, n = data_image.shape

    # Calculate the number of batches
    num_batches = len(data_image) // args.batch_size

    if args.use_umap:
        data_umap = umap.UMAP(metric = 'cosine',n_components=3,random_state = 0).fit_transform(data_image)
        data_umap = MinMaxScaler().fit_transform(data_umap)
        data_umap = torch.from_numpy(data_umap.astype(np.float32))
        loss_func2 = torch.nn.MSELoss()

        if args.use_gpu == True:
            data_umap = data_umap.cuda()

    data_image = torch.from_numpy(data_image.astype(np.float32))

    if args.use_gpu == True:
        data_image = data_image.cuda()

    model = Autoencoder(n)


    model = Autoencoder(n)
    if pretrain_path is not None and os.path.isfile(pretrain_path):
        model.load_state_dict(torch.load(pretrain_path))
        for param in model.parameters():
            param.requires_grad = True


    # optimizer = torch.optim.SGD(autoencoder.parameters(), lr=args.lr_dr,momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_dr)
    loss_func1 = CosineLoss()

    with open(args.output_file + '.csv', 'w', newline='') as csvfile:
        # Create a CSV writer
        writer = csv.writer(csvfile)
        # Write the header row
        writer.writerow(['Epoch', 'Loss'])

        for epoch in range(args.epoch_dr):
        #     if args.use_gpu == True:
        #         model = model.cuda()
        #
        #     encoded, deconder = model(data_image)
        #
        #     if args.use_umap == True:
        #
        #         loss = loss_func2(encoded, data_umap) + loss_func1(deconder, data_image)
        #
        #     else:
        #
        #         loss = loss_func1(deconder, data_image)
        #
        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()
        #     print('epoch', epoch, '| train loss:  %.4f' % loss.data.cpu().numpy())
            for i in range(num_batches):
                # Get the batch data
                batch_data = data_image[i * args.batch_size:(i + 1) * args.batch_size]
                batch_data_umap = data_umap[i * args.batch_size:(i + 1) * args.batch_size]

                if args.use_gpu == True:
                    batch_data = batch_data.cuda()
                    batch_data_umap = batch_data_umap.cuda()
                    model = model.cuda()

                encoded, deconder = model(batch_data)

                if args.use_umap == True:
                    loss = loss_func2(encoded, batch_data_umap) + loss_func1(deconder, batch_data)
                else:
                    loss = loss_func1(deconder, batch_data)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if args.use_gpu == True:
                    torch.cuda.empty_cache()

            # Write a row to the CSV file
            writer.writerow([epoch, loss.data.cpu().numpy()])

            print('epoch', epoch, '| train loss:  %.4f' % loss.data.cpu().numpy())

    torch.save(model.state_dict(), 'FEmodule_weight.pth')

    if args.use_gpu == True:
        torch.cuda.empty_cache()


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.kaiming_uniform(m.weight, mode='fan_in')
        # nn.init.constant_(m.bias, 0)



def logisticRegression(output, output2, csv_path, patch_size):
    # Extract patches from the output feature maps
    output_patches = output.unfold(1, patch_size, 1).unfold(2, patch_size, 1)
    output2_patches = output2.unfold(1, patch_size, 1).unfold(2, patch_size, 1)

    # Flatten the patches
    output_patches_flattened = output_patches.reshape(output_patches.shape[0], -1)
    output2_patches_flattened = output2_patches.reshape(output2_patches.shape[0], -1)

    # Load the CSV file
    df = pd.read_csv(csv_path)

    # Create a dictionary where the keys are the (X, Y) coordinates and the values are the class labels
    csv_dict = {(row['X'], row['Y']): row for _, row in df.iterrows()}

    # Initialize the features and labels
    features = []
    labels = []


    # For each patch
    for y in range(0, output.shape[0] - patch_size + 1):
        for x in range(0, output.shape[1] - patch_size + 1):
            # Get the class labels for the corresponding region in the CSV file
            patch_labels = [csv_dict.get((x + dx, y + dy))['class'] for dx in range(patch_size) for dy in range(patch_size) if csv_dict.get((x + dx, y + dy)) is not None]

            # If there are any class labels for the corresponding region in the CSV file
            if patch_labels:
                # Get the most common class label in the region
                most_common_label = stats.mode(patch_labels)[0][0]

                # Add the feature vector for the pixel and the corresponding m/z values to the features
                features.append(np.concatenate([output_patches_flattened[y * output.shape[1] + x, :], output2_patches_flattened[y * output.shape[1] + x, :], df.loc[(df['X'] == x) & (df['Y'] == y), 'm/z'].values]))

                # Add the most common class label to the labels
                labels.append(most_common_label)


    # Encode the class labels as integers
    le = LabelEncoder()
    labels = le.fit_transform(labels)

    # Split the features and labels into a training set and a test set
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2,
                                                                                random_state=42)
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

def Feature_Clustering(image, args, pretrain_path=None, pretrain_path2=None):
    print('Step 2: Feature Clustering ...')
    m, n, k = args.input_shape

    image = image.reshape(m, n, k)

    im = torch.from_numpy(np.array([image.transpose((2, 0, 1)).astype('float32')]))
    if args.use_gpu == True:
        im = im.cuda()

    label_colours = np.random.randint(255, size=(args.nChannel, 3))
    if args.mode == 'spat-contig':
        model = FCNetwork_SPAT_spec(k, args)
        model2 = FCNetwork_SPAT_spec(k, args)
        modelAverage = FCNetwork_SPAT_spec(k, args)
        model2Average = FCNetwork_SPAT_spec(k, args)

    if args.mode == 'spat-spor':
        model = FCNetwork_spat_SPEC(k, args)
        model2 = FCNetwork_spat_SPEC(k, args)
        modelAverage = FCNetwork_spat_SPEC(k, args)
        model2Average = FCNetwork_spat_SPEC(k, args)

    if pretrain_path is not None and os.path.isfile(pretrain_path):
        model.load_state_dict(torch.load(pretrain_path))
        for param in model.parameters():
            param.requires_grad = True
    if pretrain_path2 is not None and os.path.isfile(pretrain_path2):
        model2.load_state_dict(torch.load(pretrain_path2))
        for param in model2.parameters():
            param.requires_grad = True

    model.train()
    model.apply(weight_init)

    model2.train()
    model2.apply(weight_init)

    pretrain = torch.load('FEmodule_weight.pth')

    model_dict = model.state_dict()
    pretrain = {k: v for k, v in pretrain.items() if k in model_dict.keys()}
    model_dict.update(pretrain)

    model_dict2 = model2.state_dict()
    pretrain = {k: v for k, v in pretrain.items() if k in model_dict2.keys()}
    model_dict2.update(pretrain)

    model.load_state_dict(model_dict)
    model2.load_state_dict(model_dict2)

    HPy_target = torch.zeros(image.shape[0] - 1, image.shape[1], args.nChannel)
    HPz_target = torch.zeros(image.shape[0], image.shape[1] - 1, args.nChannel)

    if args.use_gpu == True:
        HPy_target = HPy_target.cuda()
        HPz_target = HPz_target.cuda()

    loss_fn = torch.nn.CrossEntropyLoss()
    loss_soft = TripletLoss()
    loss_fn2 = torch.nn.CrossEntropyLoss()

    loss_hpy = torch.nn.L1Loss(size_average=True)
    loss_hpz = torch.nn.L1Loss(size_average=True)

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr_fc, momentum=args.momentum_fc)
    optimizer2 = optim.SGD(filter(lambda p: p.requires_grad, model2.parameters()), lr=args.lr_fc, momentum=args.momentum_fc)

    # model.parameter = modelAverage.parameter
    for modelPara, modelAvePara in zip(model.parameters(), modelAverage.parameters()):
        modelAvePara.data = modelPara.data
    for modelPara, modelAvePara in zip(model2.parameters(), model2Average.parameters()):
        modelAvePara.data = modelPara.data

    a = []

    with open(args.output_file + 'ari.csv', 'w', newline='') as csvfile:
        # Create a CSV writer
        writer = csv.writer(csvfile)
        # Write the header row
        writer.writerow(['Epoch', 'nLabels', 'Loss', 'ARI'])

        for batch_idx in range(args.epoch_fc):
            # forwarding
            # initial parameter and get four output result

            optimizer.zero_grad()
            optimizer2.zero_grad()
            if args.use_gpu == True:
                model.cuda()
                model2.cuda()
                modelAverage.cuda()
                model2Average.cuda()


            outputq, _,xxxx1 = model(im, m, n, k)
            output2q, _,xxxx2 = model2(im, m, n, k)
            outputAverageq, _, xxxx3 = modelAverage(im, m, n, k)
            output2Averageq, xxx,xxxx4 = model2Average(im, m, n, k)

            # deal with ouput (response maps)
            output = outputq.permute(1, 2, 0).contiguous().view(-1, args.nChannel)
            output2 = output2q.permute(1, 2, 0).contiguous().view(-1, args.nChannel)

            outputAverage = outputAverageq.permute(1, 2, 0).contiguous().view(-1, args.nChannel)
            ignore, Averagetarget = torch.max(outputAverage, 1)
            im_Averagetarget = Averagetarget.data.cpu().numpy()

            #im_Averagetarget, im_Average2target are the cluster labels
            output2Average = output2Averageq.permute(1, 2, 0).contiguous().view(-1, args.nChannel)
            ignore, Average2target = torch.max(output2Average, 1)
            im_Average2target = Average2target.data.cpu().numpy()
            ari = adjusted_rand_score(im_Average2target,im_Averagetarget)

            print('ARI is : '+ str(ari))

            # initail tv loss
            outputHP = output.reshape((image.shape[0], image.shape[1] ,  args.nChannel))
            HPy = outputHP[1:, :, :] - outputHP[0:-1, :, :]
            HPz = outputHP[:, 1:, :] - outputHP[:, 0:-1, :]
            lhpy = loss_hpy(HPy, HPy_target)
            lhpz = loss_hpz(HPz, HPz_target)

            outputHP2 = output2.reshape((image.shape[0] , image.shape[1] , args.nChannel))
            HPy2 = outputHP2[1:, :, :] - outputHP2[0:-1, :, :]
            HPz2 = outputHP2[:, 1:, :] - outputHP2[:, 0:-1, :]
            lhpy2 = loss_hpy(HPy2, HPy_target)
            lhpz2 = loss_hpz(HPz2, HPz_target)

            ignore, target = torch.max(output, 1)
            target_latest = target

            ignore, target2 = torch.max(output2, 1)
            target2_latest = target2

            nLabels = len(np.unique(im_Averagetarget))

            if batch_idx % 3 == 0:

                im_target2_rgb = np.array([label_colours[c % args.nChannel] for c in im_Averagetarget])
                im_target2_rgb = im_target2_rgb.reshape(m, n, 3).astype(np.uint8)

                im_target_rgb22 = np.array([label_colours[c % args.nChannel] for c in im_Average2target])
                im_target_rgb22 = im_target_rgb22.reshape(m, n, 3).astype(np.uint8)

                cv2.imwrite(args.output_file + ".jpg", im_target2_rgb)

            if args.mode == 'spat-spor':
                args.stepsize_tv = 0

            loss1 = args.stepsize_tv * (lhpy + lhpz) + args.stepsize_sta * loss_soft(output, Average2target,batch_idx) \
                    + args.stepsize_sim*loss_fn(output, target_latest)

            loss2 = args.stepsize_tv * (lhpy2 + lhpz2) + args.stepsize_sta * loss_soft(output2, Averagetarget,batch_idx) \
                    + args.stepsize_sim*loss_fn(output2, target2_latest)

            loss = loss1 + loss2

            loss.backward(retain_graph=True)
            optimizer.step()
            optimizer2.step()

            for modelPara, modelAvePara in zip(model.parameters(), modelAverage.parameters()):
                modelAvePara.data = args.meanNetStep * modelPara.data + (1 - args.meanNetStep) * modelAvePara.data

            for modelPara, modelAvePara in zip(model2.parameters(), model2Average.parameters()):
                modelAvePara.data = args.meanNetStep * modelPara.data + (1 - args.meanNetStep) * modelAvePara.data

            # Write a row to the CSV file
            writer.writerow([batch_idx, nLabels, loss.item(), ari])

            print(batch_idx, '/', args.epoch_fc, ':', nLabels, loss2.item())

            if nLabels <= 2:
                break

            torch.save(model.state_dict(), 'FCmodule_weight.pth')
            torch.save(model2.state_dict(), 'FC2module_weight.pth')

            # Assume that `output` and `output2` are your feature maps
            np.save('output.npy', output.cpu().detach().numpy())
            np.save('output2.npy', output2.cpu().detach().numpy())
            np.save('outputAverage.npy', outputAverage.cpu().detach().numpy())

        classify = logisticRegression(output, output2, args.labels, 5)

    return im_Average2target
