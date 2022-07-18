from tensorflow import keras
import numpy as np
from osgeo import gdal


class Water3Ch(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]
        
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        
        for j, path in enumerate(batch_input_img_paths):
            
            img_arr = gdal.Open(path).ReadAsArray()
            img = np.moveaxis(img_arr, 0, -1)

            if img.shape[2] == 4:
                img = img[:,:,:3]

            x[j] = img
            
        y = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="uint8")
        for j, path in enumerate(batch_target_img_paths):
            img = gdal.Open(path).ReadAsArray()
            img = img[2,:,:] # Layer 0: Rice/Water, Layer 1:Rice/Soya, Layer 2: Water
            mask = img.copy()
            img[mask==255.] = 1
            img[mask!=255.] = 0
            
            y[j] = np.expand_dims(img, 2)
        return x, y

class Rice3Ch(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]
        
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        
        for j, path in enumerate(batch_input_img_paths):
            
            img_arr = gdal.Open(path).ReadAsArray()
            img = np.moveaxis(img_arr, 0, -1)

            if img.shape[2] == 4:
                img = img[:,:,:3]

            x[j] = img
            
        y = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="uint8")
        for j, path in enumerate(batch_target_img_paths):
            img = gdal.Open(path).ReadAsArray()
            img = img[0,:,:] # Layer 0: Rice/Water, Layer 1:Rice/Soya, Layer 2: Water
            mask = img.copy()
            img[mask==255.] = 1
            img[mask!=255.] = 0
            
            y[j] = np.expand_dims(img, 2)
        return x, y

class Orange3Ch(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]
        
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        
        for j, path in enumerate(batch_input_img_paths):
            
            img_arr = gdal.Open(path).ReadAsArray()
            img = np.moveaxis(img_arr, 0, -1)

            if img.shape[2] == 4:
                img = img[:,:,:3]

            x[j] = img
            
        y = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="uint8")
        for j, path in enumerate(batch_target_img_paths):
            img = gdal.Open(path).ReadAsArray()
            img = img[0,:,:] # Layer 0: Rice/Orange, Layer 1:Rice/Soya, Layer 2: Water
            mask = img.copy()
            img[mask==255.] = 1
            img[mask!=255.] = 0
            
            y[j] = np.expand_dims(img, 2)
        return x, y

class Cana3Ch(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]
        
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        
        for j, path in enumerate(batch_input_img_paths):
            
            img_arr = gdal.Open(path).ReadAsArray()
            img = np.moveaxis(img_arr, 0, -1)

            if img.shape[2] == 4:
                img = img[:,:,:3]

            x[j] = img
            
        y = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="uint8")
        for j, path in enumerate(batch_target_img_paths):
            img = gdal.Open(path).ReadAsArray()
            img = img[1,:,:] # Layer 0: Rice/Orange, Layer 1:Rice/Soya/Cana, Layer 2: Water
            mask = img.copy()
            img[mask==255.] = 1
            img[mask!=255.] = 0
            
            y[j] = np.expand_dims(img, 2)
        return x, y

class Rice1Ch(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]
        
        x = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="float32")
        
        for j, path in enumerate(batch_input_img_paths):
            
            img_arr = gdal.Open(path).ReadAsArray()

            x[j] = np.expand_dims(img_arr, 2)
            
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        for j, path in enumerate(batch_target_img_paths):
            img = gdal.Open(path).ReadAsArray()
            img = img[0,:,:]
            mask = img.copy()
            img[mask==255.] = 1
            img[mask!=255.] = 0
            
            y[j] = np.expand_dims(img, 2)
        return x, y