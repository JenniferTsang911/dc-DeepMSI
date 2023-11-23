from config import get_arguments
import os
from trainer import *
import argparse
import pandas as pd

parser = get_arguments()

parser.add_argument('--input_file',required= True, help = 'path to inputting msi data')
parser.add_argument('--input_shape',type = int, nargs = '+', help='input file shape',)
parser.add_argument('--mode',
                    help = 'spat-contig mode for Spatially contiguous ROI, spat-spor for Spatially sporadic ROI',
                    default= 'spat-contig')
parser.add_argument('--output_file', default='output',help='output file name')

parser.add_argument('--state_dict_auto_weights', required= False, help = 'path to autoencoder weights')
parser.add_argument('--state_dict_FC_weights', required= False, help = 'path to model weights')
parser.add_argument('--state_dict_FC_weights2', required= False, help = 'path to model2 weights')
parser.add_argument('--FE_images', nargs = '+', required= False, help = 'path to FE train dataset')
parser.add_argument('--batch_size', type = int, default= 1, help = 'batch size')
parser.add_argument('--labels', required= False, help = 'labels for FE train dataset')

if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    args = parser.parse_args()

    # data_image_df = pd.read_csv(args.input_file)
    # data_image_df.fillna(0, inplace=True)
    # data_image = data_image_df.values

    #datasets = args.input_file

    data_image = np.loadtxt(args.input_file)

    #FE_images = [np.loadtxt(dataset_path) for dataset_path in args.FE_images]
    #FE_images = np.concatenate(FE_images, axis=0)
    #print(f'Concatenated shape: {FE_images.shape}')

    #Dimension_Reduction(data_image, args, args.state_dict_auto_weights)

    im_Average2target = Feature_Clustering(data_image, args, args.state_dict_FC_weights, args.state_dict_FC_weights2)

    #np.savetxt(args.output_file + '.txt', im_Average2target)

