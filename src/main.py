import argparse
import json
import os
import glob
from tqdm import tqdm
import random
from utils import utils
from utils.model import Rice3Ch

if __name__ == '__main__':

    input_dir = "/Users/matheus/Documents/datasets/crop_types/train_rice/tiles/"
    target_dir = "/Users/matheus/Documents/datasets/crop_types/train_rice/labels/"
    model_name = "croptype_rice_rgb_jan2021"
    img_size = (256, 256)
    num_classes = 2
    batch_size = 32
    val_samples = 512
    nch = 3

    input_img_paths = sorted(
        [
            os.path.join(input_dir, fname)
            for fname in os.listdir(input_dir)
            if fname.endswith(".tif")
        ]
    )
    target_img_paths = sorted(
        [
            os.path.join(target_dir, fname)
            for fname in os.listdir(target_dir)
            if fname.endswith(".tif") and not fname.startswith(".")
        ]
    )

    assert len(input_img_paths) == len(target_img_paths), "Oh no! {} of images is different from {} of labels".format(len(input_img_paths), len(target_img_paths))

    # Build model
    model = utils.get_unet(dropout = 0.20, batchnorm = True, n_channels=nch)

    # Split img paths into a training and a validation set
    random.Random(1337).shuffle(input_img_paths)
    random.Random(1337).shuffle(target_img_paths)

    train_input_img_paths = input_img_paths[:-val_samples]
    train_target_img_paths = target_img_paths[:-val_samples]

    val_input_img_paths = input_img_paths[-val_samples:]
    val_target_img_paths = target_img_paths[-val_samples:]

    # Instantiate data Sequences for each split
    train_gen = Rice3Ch(
        batch_size, img_size, train_input_img_paths, train_target_img_paths
    )

    val_gen = Rice3Ch(
        batch_size, img_size, val_input_img_paths, val_target_img_paths
    )

    print("\nStart training... \n")
    utils.train_unet(model, model_name, train_gen, val_gen, 50)

    print("\nModel trained \n")