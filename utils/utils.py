import imp
import sys
import os
import numpy as np
from scipy import misc
import time
import math
import random
from datetime import datetime
import shutil

# Compare between every row of x1 and x2
def cosine_pair(x1, x2):
    assert x1.shape == x2.shape
    epsilon = 1e-10
    x1_norm = np.sqrt(np.sum(np.square(x1), axis=1, keepdims=True))
    x2_norm = np.sqrt(np.sum(np.square(x2), axis=1, keepdims=True))
    x1 = x1 / (x1_norm+epsilon)
    x2 = x2 / (x2_norm+epsilon)
    dist = np.sum(x1 * x2, axis=1)
    return dist

def save_manifold(images, path):
    images = (images+1.) / 2
    manifold_size = image_manifold_size(images.shape[0])
    manifold_image = np.squeeze(merge(images, manifold_size))
    misc.imsave(path, manifold_image)
    return manifold_image

def imresize(images):
    n = []
    for i in images:
        n.append( (scipy.misc.imresize(i, (160, 160, 3)) - 127.5) / 128.0)
    return np.array(n)

def image_manifold_size(num_images):
    manifold_h = int(np.floor(np.sqrt(num_images)))
    manifold_w = int(np.ceil(np.sqrt(num_images)))
    assert manifold_h * manifold_w == num_images
    return manifold_h, manifold_w

def merge(images, size):
    h, w, c = tuple(images.shape[1:4])
    manifold_image = np.zeros((h * size[0], w * size[1], c))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        manifold_image[j * h:j * h + h, i * w:i * w + w, :] = image
    if c == 1:
        manifold_image = manifold_image[:,:,:,0]
    return manifold_image

def import_file(full_path_to_module, name='module.name'):
    
    module_obj = imp.load_source(name, full_path_to_module)
    
    return module_obj

def create_log_dir(config, config_file):
    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    log_dir = os.path.join(os.path.expanduser(config.log_base_dir), config.name, subdir)
    if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
        os.makedirs(log_dir)
    shutil.copyfile(config_file, os.path.join(log_dir,'config.py'))

    return log_dir

def get_updated_learning_rate(global_step, config):
    if config.learning_rate_strategy == 'step':
        max_step = -1
        learning_rate = 0.0
        for step, lr in config.learning_rate_schedule.items():
            if global_step >= step and step > max_step:
                learning_rate = lr
                max_step = step
        if max_step == -1:
            raise ValueError('cannot find learning rate for step %d' % global_step)
    elif config.learning_rate_strategy == 'cosine':
        initial = config.learning_rate_schedule['initial']
        interval = config.learning_rate_schedule['interval']
        end_step = config.learning_rate_schedule['end_step']
        step = math.floor(float(global_step) / interval) * interval
        assert step <= end_step
        learning_rate = initial * 0.5 * (math.cos(math.pi * step / end_step) + 1)
    return learning_rate

def display_info(epoch, step, duration, watch_list):
    sys.stdout.write('[%d][%d] time: %2.2f' % (epoch+1, step+1, duration))
    for item in watch_list.items():
        if type(item[1]) in [float, np.float32, np.float64]:
            sys.stdout.write('   %s: %2.3f' % (item[0], item[1]))
        elif type(item[1]) in [int, bool, np.int32, np.int64, np.bool]:
            sys.stdout.write('   %s: %d' % (item[0], item[1]))
    sys.stdout.write('\n')
