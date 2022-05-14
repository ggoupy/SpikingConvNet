import numpy as np
from scipy.ndimage import correlate
from math import ceil
from mnist import MNIST


# Some functions are adapted from https://github.com/npvoid/SDNN_python



def spike_encoding(img, nb_timesteps):
    """
    Encode an image into spikes using a temporal coding based on pixel intensity.
    
    Args : 
        img (ndarray) : input of shape (height,width)
        nb_timesteps (int) : number of spike bins
    """
    # Intensity to latency
    with np.errstate(divide='ignore',invalid='ignore'): # suppress dive by zero warning  
        I, lat = np.argsort(1/img.flatten()), np.sort(1/img.flatten())
    # Remove pixels of value 0
    I = np.delete(I, np.where(lat == np.inf))
    # Convert 1D into 2D coordinates
    II = np.unravel_index(I, img.shape)
    # Compute the number of steps
    t_step = np.ceil(np.arange(I.size) / (I.size / (nb_timesteps-1))).astype(np.uint8)
    # Add dimension axis to index array
    # shape : (timestep, height, width)
    II = (t_step,) + II
    # Create spikes
    spike_times = np.zeros((nb_timesteps, img.shape[0], img.shape[1]), dtype=np.uint8)
    spike_times[II] = 1
    return spike_times



def DoG_filter(img, filt, threshold):
    """
    Apply a DoG filter on the given image. 
    
    Args : 
        img (ndarray) : input of shape (height,width)
        filt (ndarray) : DoG filter
        threshold (int) : threshold applied on contrasts
    """
    # Apply filter on input image
    img = correlate(img, filt, mode='constant')
    # Set to 0 borders
    border = np.zeros(img.shape)
    border[5:-5, 5:-5] = 1.
    img = img * border
    # Keep pixels bigger than the threshdold
    img = (img >= threshold).astype(int) * img
    img = np.abs(img)
    return img



def DoG(size, s1, s2):
    """
    Create a DoG filter. 
    
    Args : 
        size (int) : size of the filter
        s1 (int) : std1
        s2 (int) : std2
    """
    r = np.arange(size)+1
    x = np.tile(r, [size, 1])
    y = x.T
    d2 = (x-size/2.-0.5)**2 + (y-size/2.-0.5)**2
    filt = 1/np.sqrt(2*np.pi) * (1/s1 * np.exp(-d2/(2*(s1**2))) - 1/s2 * np.exp(-d2/(2*(s2**2))))
    filt -= np.mean(filt[:])
    filt /= np.amax(filt[:])
    return filt



def preprocess_MNIST(dataset, nb_timesteps, filters, threshold):
    """
    Preprocess the MNIST dataset. 
    """
    nb_channels = len(filters)
    samples, height, width = dataset.shape
    out = np.zeros((samples, nb_timesteps, nb_channels, height, width), dtype=np.uint8)
    for i,img in enumerate(dataset):
        encoded_img = np.zeros((nb_channels, nb_timesteps, height, width))
        for f,filt in enumerate(filters):
            dog_img = DoG_filter(img, filt, threshold)
            encoded_img[f] = spike_encoding(dog_img, nb_timesteps)
        out[i] = np.swapaxes(encoded_img,0,1)
    return out



def load_MNIST(data_prop=1):
    """
    Load the MNIST dataset. 
    """
    mndata = MNIST()
    images, labels = mndata.load_training()
    
    # Training set
    X_train, y_train = np.asarray(images), np.asarray(labels)
    if data_prop < 1:
        samples_ind = np.random.choice(len(X_train), int(len(X_train)*data_prop), replace=False)
        X_train = X_train[samples_ind]
        y_train = y_train[samples_ind]
    X_train = X_train.reshape(-1, 28, 28)
    # Random shuffling
    random_indices = np.random.permutation(len(X_train))
    X_train, y_train = X_train[random_indices], y_train[random_indices]

    # Testing set
    images, labels = mndata.load_testing()
    X_test, y_test = np.asarray(images), np.asarray(labels)
    if data_prop < 1:
        samples_ind = np.random.choice(len(X_test), int(len(X_test)*data_prop), replace=False)
        X_test = X_test[samples_ind]
        y_test = y_test[samples_ind]
    X_test = X_test.reshape(-1, 28, 28)

    input_shape = X_test[0].shape

    return X_train, y_train, X_test, y_test, input_shape



def load_encoded_MNIST(data_prop=1, nb_timesteps=15, threshold=15, filters=[DoG(7,1,2),DoG(7,2,1)]):
    """
    Load and preprocess the MNIST dataset. 
    """
    X_train, y_train, X_test, y_test, _ = load_MNIST(data_prop)
    X_train_encoded = preprocess_MNIST(X_train, nb_timesteps, filters, threshold)
    X_test_encoded = preprocess_MNIST(X_test, nb_timesteps, filters, threshold)
    return X_train_encoded, y_train, X_test_encoded, y_test