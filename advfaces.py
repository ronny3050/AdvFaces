"""Main Training File for AdvFaces Generator
"""
# MIT License
# 
# Copyright (c) 2019 Debayan Deb
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import tensorflow as tf
import numpy as np
import os

class AdvFaces:
    def __init__(self):
        self.graph = tf.Graph()
        gpu_options = tf.GPUOptions(allow_growth=True)
        tf_config = tf.ConfigProto(gpu_options=gpu_options,
                allow_soft_placement=True, log_device_placement=False)
        self.sess = tf.Session(graph=self.graph, config=tf_config)

    def generate_images(self, images, targets=None, batch_size=128, return_targets=False):
        num_images = images.shape[0]
        h,w,c = tuple(self.G.shape[1:])
        result = np.ndarray((num_images, h,w,c), dtype=np.float32)
        perturb = np.ndarray((num_images, h,w,c), dtype=np.float32)
        for start_idx in range(0, num_images, batch_size):
            end_idx = min(num_images, start_idx + batch_size)
            im = images[start_idx:end_idx]
            feed_dict = {
                    self.images: im,
                    self.phase_train: False,
                    self.keep_prob: 1.0
                }
            result[start_idx:end_idx], perturb[start_idx:end_idx] = self.sess.run([self.G,self.perturb], feed_dict=feed_dict)
        return result, perturb

    def load_model(self, model_path):
        print('LOADING MODEL FROM {}'.format(model_path))
        with self.sess.graph.as_default():
            model_path = os.path.expanduser(model_path)
            # Load grapha and variables separatedly.
            meta_files = [file for file in os.listdir(model_path) if file.endswith('.meta')]
            assert len(meta_files) == 1
            meta_file = os.path.join(model_path, meta_files[0])
            ckpt_file = tf.train.latest_checkpoint(model_path)
            
            print('Metagraph file: %s' % meta_file)
            print('Checkpoint file: %s' % ckpt_file)
            saver = tf.train.import_meta_graph(meta_file, clear_devices=True, import_scope=None)
            saver.restore(self.sess, ckpt_file)

            #print([n.name for n in self.graph.as_graph_def().node if 'Generator' in n.name])
            self.phase_train = self.graph.get_tensor_by_name('phase_train:0')
            self.keep_prob = self.graph.get_tensor_by_name('keep_prob:0')
            self.perturb = self.graph.get_tensor_by_name('Generator_1/output:0')
            self.G = self.graph.get_tensor_by_name('Generator_1/sub:0')
            self.images = self.graph.get_tensor_by_name('images:0')