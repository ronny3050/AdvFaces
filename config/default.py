''' Config Proto '''

import sys
import os


####### INPUT OUTPUT #######

# The name of the current model for output
name = 'default'

# The folder to save log and model
log_base_dir = 'log'

facenet_model_dir = 'pretrained/facenet'

# Whether to save the model checkpoints and result logs
save_model = True

# The interval between writing summary
summary_interval = 100

gan_version = 'wgan'

disc_counter = 5

# Prefix to the image files
os.environ['datasets'] = '../../adversarial-face/'

# Training data list
train_dataset_path = os.environ['datasets'] + 'CASIA-WebFace_aligned_160'

# Test data list
test_dataset_path = os.environ['datasets'] + 'LFW_aligned_160'

# Target image size (h,w) for the input of network
image_size = (160,160)

# 3 channels means RGB, 1 channel for grayscale
channels = 3

gradient_clip = True

batch_format='random_samples'

z_dim = 128

# Preprocess for training
preprocess_train = [
    ['standardize', 'mean_scale'],
]

# Preprocess for testing
preprocess_test = [
    ['resize', image_size],
    ['standardize', 'mean_scale'],
]

# Number of GPUs
num_gpus = 1


####### NETWORK #######

# The network architecture
network = 'nets/debgan.py'

# Dimensionality of the bottleneck layer in discriminator
bottleneck_size = 512

# Dimensionality of the style space
style_size = 8

target_model = 'facenet'

####### TRAINING STRATEGY #######

# Optimizer
optimizer = ("ADAM", {'beta1': 0.5, 'beta2': 0.9})
# optimizer = ("MOM", {'momentum': 0.9})

# Number of samples per batch
batch_size = 32

# Number of batches per epoch
epoch_size = 200

# Number of epochs
num_epochs = 500

# learning rate strategy
learning_rate_strategy = 'step'

# learning rate schedule
lr = 0.1
learning_rate_schedule = {
    0: 1 * lr,
}

learning_rate_multipliers = {}

# Restore model
restore_model = None

# Keywords to filter restore variables, set None for all
restore_scopes = None

# Weight decay for model variables
weight_decay = 0.0

# Keep probability for dropouts
keep_prob = 1.0

####### LOSS FUNCTION #######

# Weight of the global adversarial loss
coef_adv = 1.0

# Weight of the patch adversarial loss
coef_patch_adv = 2.0

# Weight of the identity mapping loss
coef_idt = 10.0

pixel_loss_factor = 1.0

idt_loss_factor = 10.0
