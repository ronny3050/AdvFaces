from utils.imageprocessing import preprocess
from utils.dataset import Dataset
from utils import utils
from advfaces import AdvFaces

import os
import scipy.misc

# Load Adversarial Face Generator
network = AdvFaces()
network.load_model('pretrained/obfuscation')

## Load images
# Images can be loaded via
# 1. Image Filelist
# dataset = Dataset('image_list.txt')

# 2. Folder of images
dataset = Dataset('data')

# Load config and images
config = utils.import_file('config/default.py', 'config')
images = preprocess(dataset.images, config, is_training=False)

# Generate Adversarial Images and Adversarial Masks
adversaries, adversaries_mask = network.generate_images(images)

# Save adversarial image
scipy.misc.imsave('results/result.jpg', adversaries[0])