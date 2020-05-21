''' Config Proto '''

import sys
import os

#############################
####### TRAINING DETAILS ####
#############################
# Training for obfuscation or target settings
mode = 'obfuscation'
# Prefix to the image files
os.environ['datasets'] = '/home/debayan/Research/adversarial-face/'
# Training path
train_dataset_path = os.environ['datasets'] + 'CASIA-WebFace_aligned_160'
# Test path
test_dataset_path = os.environ['datasets'] + 'LFW_aligned_160'

####### AUXILLIARY MATCHER ######
aux_matcher_definition = 'architectures/inception_resnet_v1.py'
aux_matcher_path = 'pretrained/facenet/model-20180402-114759.ckpt-275'
aux_matcher_scope = 'InceptionResnetV1'
# Matching Threshold. !!!!  CAREFUL -- By default, we assume scores are un-normalized between [-1, 1]
aux_matcher_threshold = 0.45	

####### LOSS FUNCTION #######
pixel_loss_factor = 1.0
perturb_loss_factor = 1.0
idt_loss_factor = 10.0
MAX_PERTURBATION = 3.0

# learning rate strategy
learning_rate_strategy = 'step'

# learning rate schedule
lr = 0.0001
learning_rate_schedule = {
    0: 1 * lr,
}
learning_rate_multipliers = {}
# Number of samples per batch
batch_size = 32

# Number of batches per epoch
epoch_size = 200

# Number of epochs
num_epochs = 500

#############################
####### MODEL DETAILS ######
#############################
# The name of the current model for output
name = 'default'

# The folder to save log and model
log_base_dir = 'log'

# Whether to save the model checkpoints and result logs
save_model = True

# Target image size (h,w) for the input of network
image_size = (160, 160)

# 3 channels means RGB, 1 channel for grayscale
channels = 3

# The interval between writing summary
summary_interval = 100

gan_version = 'wgan'

disc_counter = 5

gradient_clip = True

batch_format='random_samples'

z_dim = 128

# Preprocess for training
preprocess_train = [
	['resize', image_size],
	['random_flip'],
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
network = 'architectures/advfaces.py'

# Dimensionality of the bottleneck layer in discriminator
bottleneck_size = 512


####### TRAINING STRATEGY #######

# Optimizer
optimizer = ("ADAM", {'beta1': 0.5, 'beta2': 0.9})
# optimizer = ("MOM", {'momentum': 0.9})

# Restore model
restore_model = None

# Keywords to filter restore variables, set None for all
restore_scopes = None

# Weight decay for model variables
weight_decay = 0.0

# Keep probability for dropouts
keep_prob = 1.0