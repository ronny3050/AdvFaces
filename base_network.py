
import sys
import time
import imp
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

from utils import tfutils

class BaseNetwork:
    def __init__(self):
        self.graph = tf.Graph()
        gpu_options = tf.GPUOptions(allow_growth=True)
        tf_config = tf.ConfigProto(gpu_options=gpu_options,
                allow_soft_placement=True, log_device_placement=False)
        self.sess = tf.Session(graph=self.graph, config=tf_config)

    def load_model(self, *args, **kwargs):
        tfutils.load_model(self.sess, *args, **kwargs)
        self.phase_train_placeholder = self.graph.get_tensor_by_name('phase_train:0')
        self.keep_prob_placeholder = self.graph.get_tensor_by_name('InceptionResnetV1/Logits/Dropout/cond/dropout/keep_prob:0')
        self.inputs = self.graph.get_tensor_by_name('input:0')
        self.outputs = self.graph.get_tensor_by_name('embeddings:0')

    def extract_feature(self, images, batch_size, proc_func=None, verbose=False):
        num_images = images.shape[0] if type(images)==np.ndarray else len(images)
        num_features = self.outputs.shape[1]
        result = np.ndarray((num_images, num_features), dtype=np.float32)
        start_time = time.time()
        for start_idx in range(0, num_images, batch_size):
            if verbose:
                elapsed_time = time.strftime('%H:%M:%S', time.gmtime(time.time()-start_time))
                sys.stdout.write('# of images: %d Current image: %d Elapsed time: %s \t\r' 
                    % (num_images, start_idx, elapsed_time))
            end_idx = min(num_images, start_idx + batch_size)
            inputs = images[start_idx:end_idx]
            inputs = proc_func(inputs) if proc_func else inputs
            feed_dict = {self.inputs: inputs,
                        self.phase_train_placeholder: False,
                        self.keep_prob_placeholder: 1.0}
            result[start_idx:end_idx] = self.sess.run(self.outputs, feed_dict=feed_dict)
        return result

        
