import argparse
import json
import os
import glob
from tqdm import tqdm
from utils import utils
from tensorflow import keras
import json
import zipfile


if __name__ == '__main__':

    # print("Creating DS")
    # utils.create_dataset("/Users/matheus/Downloads/crop_DEZ_2020")
    parser = argparse.ArgumentParser(prog='trixie',
                                     description='Process and predict master dataset  \n\n'
                                                 'Obs: Will be executed in the given order.'
                                                 'Following steps available: \n'
                                                 ' 1. Process IDL files: Perform collocation of a S1 and S2 image \n'
                                                 ' 2. Create SAR image: Creates a 3 channel image if SAR features \n', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('Steps', metavar='S', type=int, nargs='*', help='steps to be performed')
    parser.add_argument('-s2c','--sen2cor', type=str, help='Use Sen2Cor to convert Level 1C to 2A')
    parser.add_argument('-c','--crop', type=str, help='Crop a scene in small pieces, receive a folder path')
    parser.add_argument('-t','--train', nargs='+', type=str, help='Train neural network. Needs crop(rice,orange), images dir and epochs')
    parser.add_argument('-e','--evaluate', nargs='+', type=str, help='Evaluate neural network. Needs predicted mask and target')
    parser.add_argument('-m','--mask', nargs='+', type=str, help='Create a mask with all predictions')
    parser.add_argument('-n','--ndvi', type=str, help='Create NDVI images based on scenes provided')

    args = parser.parse_args()

    if args.sen2cor:

        print("\nConverting Level 1C to 2A\n")

        os.system(f"/Users/matheus/Downloads/Sen2Cor-02.10.01-Darwin64/bin/L2A_Process --resolution 10 {args.sen2cor}")

    if args.crop:

        print("\nCreating dataset cropping images\n")

        utils.create_dataset(args.crop)

    if args.train:
        print("\nTraining Neural Network\n")
        
        crop_type = args.train[0]
        train_dir = args.train[1]
        epochs = int(args.train[2])

        utils.cli_train(crop_type, train_dir, epochs)

    if args.evaluate:

        print("\nEvaluating created mask\n")

        print(f"\nPrediction: {args.evaluate[0]}\Label: {args.evaluate[1]}\n")

        utils.cli_evaluate(args.evaluate[0], args.evaluate[1])

    if args.mask:

        print("\nCreating scene mask\n")

        utils.cli_prediction_mask(args.mask[0], args.mask[1], bulky=False)

    if args.ndvi:

        IN_PATH = os.path.join(args.ndvi, '')

        files = glob.glob(IN_PATH + '*.zip')

        print("\n{} files found.\n".format(len(files)))

        for n,f in enumerate(files):

            image_name = f[-26:-11]

            image_level = f.split('/')[-1].split('_')[1][-3:]
            assert image_level == 'L2A', "Processing Level not recognized. Current only L2A is supported."

            img_zip = zipfile.ZipFile(f, 'r')

            band_file_names = []

            for elem in img_zip.namelist():
                parts = elem.split('/')
                if image_level == 'L1C':
                    if len(parts) > 4 and parts[3] == 'IMG_DATA' and len(parts[-1]) > 1:
                        band_file_names.append(elem)
                elif image_level == 'L2A':
                    if len(parts) > 5 and len(parts[-1]) > 1:
                        band_file_names.append(elem)

            print(f"\nCalculating NDVI for {image_name} \n")
            utils.ndvi(band_file_names, IN_PATH, image_name, img_zip)


