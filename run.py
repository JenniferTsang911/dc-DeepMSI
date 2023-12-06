from config import get_arguments
import os
from trainer import *
import argparse
import pandas as pd
from scipy import sparse
import os

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

# parser.add_argument('--input_files', nargs='+', required=True, help='paths to inputting msi data')
# parser.add_argument('--input_shapes', nargs='+', type=int, required=True, help='input file shapes')
# parser.add_argument('--output_files', nargs= '+', default='output',help='output file name')

if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    args = parser.parse_args()

    # data_image_df = pd.read_csv(args.input_file)
    # data_image_df.fillna(0, inplace=True)
    # data_image = data_image_df.iloc[:, 4:]
    # data_image = data_image.to_numpy()

    #datasets = args.input_file

    data_image = np.loadtxt(args.input_file)

    #data_image = sparse.csr_matrix(data_image)

    Dimension_Reduction(data_image, args, args.state_dict_auto_weights)

    #im_Average2target = Feature_Clustering(data_image, args, args.state_dict_FC_weights, args.state_dict_FC_weights2,
                                           #.output_file, args.input_shape)

    #np.savetxt(args.output_file + '.txt', im_Average2target)

    # path_in = r"C:\Users\jenni\OneDrive - Queen's University\DESI project\DESI TXT colon\Arrays"
    # path_out = r"C:\Users\jenni\OneDrive - Queen's University\DESI project\DESI TXT colon\dc-DeepMSI outputs"
    #
    # input_files = ['2021 03 29 colon 0462641-2 Analyte 2 array.txt', '2021 03 30 colon 0413337-2 Analyte 6 array.txt',
    #                '2021 03 30 colon 0720931-3 Analyte 5 array.txt', '2021 03 31 colon 1258561-2 Analyte 7 array.txt']
    # output_files = ['2021 03 29 colon 0462641-2 Analyte 2', '2021 03 30 colon 0413337-2 Analyte 6',
    #                 '2021 03 30 colon 0720931-3 Analyte 5', '2021 03 31 colon 1258561-2 Analyte 7']
    #
    # input_shapes = [(250, 202, 2000), (205, 263, 2000), (211, 248, 2000), (186, 242, 2000)]
    #
    # for input_file, output_file, input_shape in zip(args.input_files, args.output_files,
    #                                                 [tuple(args.input_shapes[i:i + 3]) for i in
    #                                                  range(0, len(args.input_shapes), 3)]):
    #
    #     print(input_file, output_file, input_shape)
    #
    #     input_file = os.path.join(path_in, input_file)
    #     output_file = os.path.join(path_out, output_file)
    #
    #     data_image = np.loadtxt(input_file)
    #
    #     Dimension_Reduction(data_image, args, args.state_dict_auto_weights)
    #
    #     im_Average2target = Feature_Clustering(data_image, args, args.state_dict_FC_weights, args.state_dict_FC_weights2, output_file, input_shape)
    #
    #     np.savetxt(output_file + '.txt', im_Average2target)

