from curses.panel import top_panel
from tensorflow.keras import layers
from tensorflow import keras
import os, sys
from osgeo import gdal
import math
import numpy as np
import glob
from tqdm import tqdm
import PIL
import random
from keras import backend as K
import rasterio as rio
from utils.model import Rice3Ch, Water3Ch, Rice1Ch, Orange3Ch, Cana3Ch
from datetime import datetime
import shutil




# ===============================================================================
# U-NET NEURAL NETWORK
# ===============================================================================

def conv2d_block(input_tensor, n_filters, kernel_size = 3, batchnorm = True):
    # first layer
    x = layers.Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(input_tensor)
    if batchnorm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    
    # second layer
    x = layers.Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(x)
    if batchnorm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x

# -------------------------------------------------------------------------------

def get_unet(nClasses=2, input_height=256, input_width=256, n_filters = 16, dropout = 0.1, batchnorm = True, n_channels=3):
    
    input_img = keras.Input(shape=(input_height,input_width, n_channels))

    # contracting path
    c1 = conv2d_block(input_img, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
    p1 = layers.MaxPooling2D((2, 2)) (c1)
    p1 = layers.Dropout(dropout)(p1)

    c2 = conv2d_block(p1, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)
    p2 = layers.MaxPooling2D((2, 2)) (c2)
    p2 = layers.Dropout(dropout)(p2)

    c3 = conv2d_block(p2, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)
    p3 = layers.MaxPooling2D((2, 2)) (c3)
    p3 = layers.Dropout(dropout)(p3)

    c4 = conv2d_block(p3, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)
    p4 = layers.MaxPooling2D(pool_size=(2, 2)) (c4)
    p4 = layers.Dropout(dropout)(p4)
    
    c5 = conv2d_block(p4, n_filters=n_filters*16, kernel_size=3, batchnorm=batchnorm)
    
    # expansive path
    u6 = layers.Conv2DTranspose(n_filters*8, (3, 3), strides=(2, 2), padding='same') (c5)
    u6 = layers.concatenate([u6, c4])
    u6 = layers.Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)

    u7 = layers.Conv2DTranspose(n_filters*4, (3, 3), strides=(2, 2), padding='same') (c6)
    u7 = layers.concatenate([u7, c3])
    u7 = layers.Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)

    u8 = layers.Conv2DTranspose(n_filters*2, (3, 3), strides=(2, 2), padding='same') (c7)
    u8 = layers.concatenate([u8, c2])
    u8 = layers.Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)

    u9 = layers.Conv2DTranspose(n_filters*1, (3, 3), strides=(2, 2), padding='same') (c8)
    u9 = layers.concatenate([u9, c1], axis=3)
    u9 = layers.Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
    
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid') (c9)
    #outputs = layers.Conv2D(nClasses, (1, 1)) (c9)
    
    model = keras.Model(inputs=[input_img], outputs=[outputs])
    return model

# -------------------------------------------------------------------------------

def train_unet(model, model_name, train_gen, val_gen, epochs=10):
    path = os.path.abspath(os.path.dirname(sys.argv[0]))  + "/models/"

    # Configure the model for training.
    # We use the "sparse" version of categorical_crossentropy
    # because our target data is integers.
    opt = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=opt, loss="binary_crossentropy")

    callbacks = [
        keras.callbacks.ModelCheckpoint(path + model_name + ".h5", save_best_only=True)
    ]

    # Train the model, doing validation at the end of each epoch.
    h = model.fit(train_gen, epochs=epochs, validation_data=val_gen, callbacks=callbacks)
    model.save(path+ model_name)

# -------------------------------------------------------------------------------

def iou_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
    union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou

# -------------------------------------------------------------------------------

def logical_iou(result1,result2):
    intersection = np.logical_and(result1, result2)
    union = np.logical_or(result1, result2)
    iou_score = np.sum(intersection) / np.sum(union)
    print("IoU is %s" % iou_score)

# -------------------------------------------------------------------------------

def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)
    return dice

# ===============================================================================
# IMAGE PROCESSING  
# ===============================================================================

def create_prediction_mask(model, imgs_path, crop_type):
    ''' Receive a path contaning .tif images and crop them into 256x256 pixels and save in the same path

    Args:
        imgs_path: List containing several images for test

    Returns:

    '''
    img_name = imgs_path[1].split("/")[-1][:13]
    pred_path = imgs_path[1][:-35] + "/preds/"
    crop_type = crop_type.upper()

    # Remove old predictions and start a clean folder
    if os.path.exists(pred_path) and os.path.isdir(pred_path):
        shutil.rmtree(pred_path)

    os.makedirs(pred_path)

    # Generate empty array with all labels
    #all_images = np.empty((1, 256,256,3))
    all_images = np.zeros((len(imgs_path), 256,256,3))

    print("\nStacking images...\n")
    # Generate predictions for all images in the test set
    for i,path in enumerate(tqdm(imgs_path)):
                
        img_arr = gdal.Open(path).ReadAsArray()
        img = np.moveaxis(img_arr, 0, -1)
        img = img[np.newaxis, :, :,:3]
        #all_images = np.vstack((all_images, img))
        all_images[i,:,:,:] = img


    print("\nPredicting...\n")
    all_preds = model.predict(all_images)

    #all_preds = all_preds[1:,:,:,:]


    print("\nSaving predictions to jpg...\n")
    for i, arrim in enumerate(tqdm(all_preds)):
        mask = np.round(arrim[:,:,0], 0)
        im = PIL.Image.fromarray(mask*256)
        im = im.convert('RGB')
        im_name = pred_path + "mask_{0}_{1:05d}.jpg".format(img_name,i)
        im.save(im_name)

    input_pred_paths = sorted(
        [
            os.path.join(pred_path, fname)
            for fname in os.listdir(pred_path)
            if fname.endswith(".jpg")
        ]
    )

    final_img_pred = PIL.Image.open(input_pred_paths[0])
    i = 1

    print("\nSaving final image...\n")
    while i < len(input_pred_paths):
        
        img2append = PIL.Image.open(input_pred_paths[i])
        final_img_pred = get_concat_v(final_img_pred,img2append)

        if i == 42:
            fim = final_img_pred.copy()
            final_img_pred = PIL.Image.open(input_pred_paths[i+1])
            i = i+2
        
        elif i in list(range(85, 1847, 43)):
            fim = get_concat_h(fim,final_img_pred)
            final_img_pred = PIL.Image.open(input_pred_paths[i+1])
            i = i+2
            
        elif i == 1848:
            fim = get_concat_h(fim,final_img_pred)
            break
            
        else: i+=1
    
    fim_name = pred_path[:-6] + f"{img_name}_MASK_{crop_type}.jpg"

    # Crop image
    left = 0
    top = 0
    right = 10980
    bottom = 10980

    fim = fim.crop((left, top, right, bottom))
    fim.save(fim_name)

    del all_images
    del all_preds
    del final_img_pred
    del fim

# -------------------------------------------------------------------------------

def create_dataset(imgs_path):
    ''' Receive a path contaning .tif images and crop them into 256x256 pixels and save in the same path

    Args:
        imgs_path: Folder with tif images

    Returns:

    '''

    imgs2crop = glob.glob(imgs_path + '/*.tif')

    print(f"\n{len(imgs2crop)} image(s) found\n")

    for im in tqdm(imgs2crop):

            if not os.path.exists(im[:-4]):
                os.makedirs(im[:-4])

            get_split(im, im[:-4] + "/")

# -------------------------------------------------------------------------------

def get_split(fileIMG, out_path):
    ''' Crop the given image into 256x256 pixels and save in out_path

    Args:
        fileIMG: A single itf file image
        out_path: Folder to save the tiles

    Returns:

    '''

    dataset = gdal.Open(fileIMG)

    passo = 256
    xsize = 1 * passo
    ysize = 1 * passo

    extent = get_extent(dataset)
    cols = int(extent["cols"])
    rows = int(extent["rows"])

    nx = (math.ceil(cols / passo))
    ny = (math.ceil(rows / passo))

    # print(nx*ny)

    cont = 0
    contp = 0

    for i in range(0, nx):
        for j in range(0, ny):
            cont += 1
            dst_dataset = out_path + os.path.basename(fileIMG)[:-4] + '_p' + str(cont).zfill(5)

            if not os.path.exists(dst_dataset):
                xoff = passo * i
                yoff = passo * j

                if xoff + xsize > cols:
                    n2 = range(xoff, cols)
                else:
                    n2 = range(xoff, xoff + xsize)

                if yoff + ysize > rows:
                    n1 = range(yoff, rows)
                else:
                    n1 = range(yoff, yoff + ysize)

                contp += 1
                gdal.Translate(dst_dataset + '.tif', dataset, srcWin=[xoff, yoff, xsize, ysize], outputType=gdal.GDT_Float32)

    return contp

# -------------------------------------------------------------------------------

def get_extent(dataset):
    cols = dataset.RasterXSize
    rows = dataset.RasterYSize
    transform = dataset.GetGeoTransform()

    minx = transform[0]
    maxx = transform[0] + cols * transform[1] + rows * transform[2]
    miny = transform[3] + cols * transform[4] + rows * transform[5]
    maxy = transform[3]

    return {"minX": str(minx), "maxX": str(maxx),
            "minY": str(miny), "maxY": str(maxy),
            "cols": str(cols), "rows": str(rows)}

# -------------------------------------------------------------------------------

def get_concat_v(im1, im2):
    dst = PIL.Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst

# -------------------------------------------------------------------------------

def get_concat_h(im1, im2):
    dst = PIL.Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

# -------------------------------------------------------------------------------

def ndvi(bands, input_path, img_name, zipf):
    """ Creates a .tif image using the Normalized Difference Vegetation Index colors

    :param bands: all bands from Sentinel-2 product
    :param input_path: location of the .zip files following the ESA SAFE format
    :param img_name: product/scene name

    """

    output_path = input_path + 'NDVI_nothres/'

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for band_name in bands:
        if band_name.endswith('_B04_10m.jp2'):
            dsRed = gdal.Open('/vsizip/%s/%s' % (zipf.filename, band_name))
            bandRed = dsRed.ReadAsArray().astype(np.float32)

        if band_name.endswith('_B08_10m.jp2'):
            dsNIR = gdal.Open('/vsizip/%s/%s' % (zipf.filename, band_name))
            geoTransform = dsNIR.GetGeoTransform()
            geoProjection = dsNIR.GetProjection()
            bandNIR = dsNIR.ReadAsArray().astype(np.float32)

    ndvi_im = np.zeros(bandRed.shape, dtype=rio.float32)

    ndvi_im = (bandNIR.astype(float) - bandRed.astype(float)) / (bandNIR + bandRed)

    # Removed for JS to play with this number
    ndvi_threshold = ndvi_im # np.where(ndvi_im > 0.7, 1, 0)

    drv = gdal.GetDriverByName("GTiff")
    dst_ds = drv.Create(output_path + img_name + '_NDVI.tif', dsNIR.RasterXSize, dsNIR.RasterYSize, 1,
                        gdal.GDT_Float32) # GDT_Byte
    dst_ds.SetGeoTransform(geoTransform)
    dst_ds.SetProjection(geoProjection)
    dst_ds.GetRasterBand(1).WriteArray(ndvi_threshold)
    dst_ds.FlushCache()

    dst_ds = None

# -------------------------------------------------------------------------------

# ===============================================================================
# COMMAND LINE INTERFACE  
# ===============================================================================

def cli_evaluate(predicted, label):

    ai = PIL.Image.open(predicted)
    label = PIL.Image.open(label)

    ar_ai = np.asarray(ai)
    ar_label = np.asarray(label)

    real_label = ar_label[:10980,:10980,0]
    maskl = real_label.copy()
    real_label[maskl==255.] = 1
    real_label[maskl!=255.] = 0

    ai_label = ar_ai[:10980,:10980,0]
    maska = ai_label.copy()
    ai_label[maska==255.] = 1
    ai_label[maska!=255.] = 0
    
    print(logical_iou(ai_label,real_label))

def cli_train(crop, train_dir, epochs, nch=3):

    if crop == "rice":
        ModelNCh = Rice3Ch
    elif crop == "orange":
        ModelNCh = Orange3Ch
    elif crop == "cana":
        ModelNCh = Cana3Ch
    elif crop == "water":
        ModelNCh = Water3Ch
    else: print(f"Couldn't find {crop} model") 
    
    input_dir = os.path.join(train_dir, "tiles/")
    target_dir = os.path.join(train_dir, "labels/")
    img_size = (256, 256)
    batch_size = 32

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

    tile_datename = input_img_paths[0].split("/")[-1][:-11]
    model_name = f"croptype_{crop}_{tile_datename}_{datetime.now().strftime('%Y%m%d')}"

    assert len(input_img_paths) == len(target_img_paths), "Oh no! {} of images is different from {} of labels".format(len(input_img_paths), len(target_img_paths))

    val_samples = 2**int(np.log2(len(input_img_paths)*0.3))

    # Build model
    model_path = os.path.abspath(os.getcwd()) + "/models/" + model_name

    if os.path.exists(model_path):
        model = keras.models.load_model(model_path)
        print(f"\nModel loaded.\n")
    else: 
        model = get_unet(dropout = 0.20, batchnorm = True, n_channels=nch)
        # REMOVE THIS REMOVE THIS REMOVE THIS VVVV
        # model.load_weights("/Users/matheus/Documents/masters/crop_type_classifier/src/models/bkp_before_trash/croptype_rice_rgb_jan2021_2days.h5")
        print(f"\nNew model created.\n")

    print(f"\nModel name: {model_name}\nTraining images: {len(input_img_paths)}\nValidation images: {val_samples}\n")

    # Split img paths into a training and a validation set
    random.Random(1337).shuffle(input_img_paths)
    random.Random(1337).shuffle(target_img_paths)

    train_input_img_paths = input_img_paths[:-val_samples]
    train_target_img_paths = target_img_paths[:-val_samples]

    val_input_img_paths = input_img_paths[-val_samples:]
    val_target_img_paths = target_img_paths[-val_samples:]

    # Instantiate data Sequences for each split
    train_gen = ModelNCh(
        batch_size, img_size, train_input_img_paths, train_target_img_paths
    )

    val_gen = ModelNCh(
        batch_size, img_size, val_input_img_paths, val_target_img_paths
    )

    print("\nStart training... \n")
    train_unet(model, model_name, train_gen, val_gen, epochs)

    print("\nModel trained \n")

    # Generate predictions for all images in the validation set

    val_gen = ModelNCh(batch_size, img_size, val_input_img_paths, val_target_img_paths)
    val_preds = model.predict(val_gen)

    print("\nEvaluating Model\n")
    # Generate array with all labels
    all_labels = np.empty((1, 256,256,1))
    #all_labels = np.zeros((len(val_gen), 256,256,1))

    for i, label in enumerate(tqdm(val_target_img_paths)):
    #     l = np.array(PIL.Image.open(label))
        l =  gdal.Open(label).ReadAsArray()[1,:,:]
        l = l[np.newaxis, :, :, np.newaxis]
        #all_labels[i,:,:,:] = l 
        all_labels = np.vstack((all_labels, l))
        
    all_labels = all_labels[1:,:].astype(np.float32)
    #all_labels = all_labels.astype(np.float32)

    round_preds = np.round(val_preds)

    iof_metric = iou_coef(round_preds, all_labels).numpy()
    dice_metric = dice_coef(round_preds, all_labels).numpy()

    print(f"\nModel achieved IoU: {iof_metric} and Dice Coef.: {dice_metric}\n")

def cli_prediction_mask(model_name, input_dir,models_dir="/models", bulky=False):

    if bulky:
        models2pred = glob.glob(os.path.abspath(os.getcwd()) + models_dir + '/*.h5')

        print(f"\n{len(models2pred)} models found\n")

        for m in models2pred:
            model_path = m[:-3]
            model_name = m.split('/')[-1][:-3]
            print(f"\nModel: {model_name}\n")

            model = keras.models.load_model(model_path)
            
            input_img_paths = sorted(
                [
                    os.path.join(input_dir, fname)
                    for fname in os.listdir(input_dir)
                    if fname.endswith(".tif")
                ]
            )
            create_prediction_mask(model, input_img_paths, model_name)
            
    else:
        print("Model manually selected")
        model_path = os.path.abspath(os.getcwd()) + f"/models/{model_name}"
        model = keras.models.load_model(model_path)
        
        print(f"\nModel: {model_name}\n")

        input_img_paths = sorted(
            [
                os.path.join(input_dir, fname)
                for fname in os.listdir(input_dir)
                if fname.endswith(".tif")
            ]
        )
        create_prediction_mask(model, input_img_paths, model_name)

    print("Done.")
    