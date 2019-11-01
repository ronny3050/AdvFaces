"""Data fetching
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

import sys
import os
import time
import math
import random
import shutil
from multiprocessing import Process, Queue

import h5py
import numpy as np

class DataClass(object):
    def __init__(self, class_name, indices, label):
        self.class_name = class_name
        self.indices = np.array(indices)
        self.label = label
        return

    def random_pair(self):
        return np.random.permutation(self.indices)[:2]

    def random_samples(self, num_samples_per_class, exception=None):
        indices_temp = list(self.indices[:])
        if exception is not None:
            indices_temp.remove(exception)
            assert len(indices_temp) > 0
        # Sample indices multiple times when more samples are required than present.
        indices = []
        iterations = int(np.ceil(1.0*num_samples_per_class / len(indices_temp)))
        for i in range(iterations):
            sample_indices = np.random.permutation(indices_temp)
            indices.append(sample_indices)
        indices = np.concatenate(indices, axis=0)[:num_samples_per_class]
        return indices

    def build_clusters(self, cluster_size):
        permut_indices = np.random.permutation(self.indices)
        cutoff = (permut_indices.size // cluster_size) * cluster_size
        clusters = np.reshape(permut_indices[:cutoff], [-1, cluster_size])
        clusters = list(clusters)
        if permut_indices.size > cutoff:
            last_cluster = permut_indices[cutoff:]
            clusters.append(last_cluster)
        return clusters

class Dataset():

    def __init__(self, path=None, mode=None):
        self.DataClass = DataClass
        self.num_classes = None
        self.classes = None
        self.images = None
        self.labels = None
        self.targets = None
        self.features = None
        self.idx2cls = None
        self.index_queue = None
        self.index_worker = None
        self.batch_queue = None
        self.batch_workers = None
        self.mode = mode

        if path is not None:
            self.init_from_path(path)

    def clear(self):
        del self.classes
        self.__init__()

    def init_from_path(self, path):
        path = os.path.expanduser(path)
        _, ext = os.path.splitext(path)
        if os.path.isdir(path):
            self.init_from_folder(path)
        elif ext == '.txt':
            self.init_from_list(path)
        elif ext == '.hdf5':
            self.init_from_hdf5(path)
        else:
            raise ValueError('Cannot initialize dataset from path: %s\n\
                It should be either a folder, .txt or .hdf5 file' % path)
        print('%d images of %d classes loaded' % (len(self.images), self.num_classes))

    def init_from_folder(self, folder):
        folder = os.path.expanduser(folder)
        class_names = os.listdir(folder)
        class_names.sort()
        images = []
        labels = []
        label = 0
        if os.path.isdir(os.path.join(folder,class_names[0])):
            for i, class_name in enumerate(class_names):
                classdir = os.path.join(folder, class_name)
                if os.path.isdir(classdir):
                    images_class = os.listdir(classdir)
                    images_class.sort()
                    images_class = [os.path.join(classdir,img) for img in images_class]
                    if len(images_class) < 2:
                        continue
                    indices_class = np.arange(len(images), len(images) + len(images_class))
                    images.extend(images_class)
                    labels.extend(len(images_class) * [label])
                    label = label + 1
            self.images = np.array(images, dtype=np.object)
            self.labels = np.array(labels, dtype=np.int32)
        else:
            self.images = np.array([os.path.join(folder, c) for c in class_names], dtype=np.object)
            self.labels = np.ones((self.images.shape[0]), dtype=np.int32)  

        self.init_classes()

    def init_from_list(self, filename):
        with open(filename, 'r') as f:
            lines = f.readlines()
        lines = [line.strip().split(' ') for line in lines]
        assert len(lines)>0, \
            'List file must be in format: "fullpath(str) label(int)"'
        images = [line[0] for line in lines]
        if len(lines[0]) > 1:
            labels = [int(line[1]) for line in lines]
        else:
            labels = [os.path.dirname(img) for img in images]
            _, labels = np.unique(labels, return_inverse=True)
        self.images = np.array(images, dtype=np.object)
        self.labels = np.array(labels, dtype=np.int32)
        self.init_classes()


    def init_from_hdf5(self, filename):
        with h5py.File(filename, 'r') as f:
            self.images = np.array(f['images'])
            self.labels = np.array(f['labels'])
        self.init_classes()
       
    def init_crossval_folder(self, folder):
        folder = os.path.expanduser(folder)
        classes = []
        images = []
        labels = []
        k_folds_classes = []
        splits = os.listdir(folder)
        splits.sort()
        for splitdir in splits:
            splitdir = os.path.join(folder, splitdir)
            fold_classes = []
            class_names = os.listdir(splitdir)
            class_names.sort()
            for label, class_name in enumerate(class_names):
                classdir = os.path.join(splitdir, class_name)
                if os.path.isdir(classdir):
                    images_class = os.listdir(classdir)
                    images_class.sort()
                    images_class = [os.path.join(classdir,img) for img in images_class]
                    indices_class = np.arange(len(images), len(images) + len(images_class))
                    images.extend(images_class)
                    labels.extend(len(images_class) * [label])
                    new_class = self.DataClass(class_name, indices_class, label)
                    classes.append(new_class)
                    fold_classes.append(new_class)
            k_folds_classes.append(fold_classes)
                    
        self.classes = np.array(classes, dtype=np.object)
        self.images = np.array(images, dtype=np.object)
        self.labels = np.array(labels, dtype=np.int32)
        self.num_classes = len(classes)
        self.k_folds_classes = k_folds_classes
        
    def init_classes(self):
        dict_classes = {}
        classes = []
        self.idx2cls = np.ndarray((len(self.labels),)).astype(np.object)    
        for i, label in enumerate(self.labels):
            if not label in dict_classes:
                dict_classes[label] = [i]
            else:
                dict_classes[label].append(i)
        for label, indices in dict_classes.items():
            classes.append(self.DataClass(str(label), indices, label))
            self.idx2cls[indices] = classes[-1]
        self.classes = np.array(classes, dtype=np.object)
        self.num_classes = len(classes)

    def import_features(self, listfile, features):
        assert self.images.shape[0] == features.shape[0]
        with open(listfile, 'r') as f:
            images = f.readlines()
        img2idx = {}
        for i, image in enumerate(images):
            img2idx[os.path.abspath(image.strip())] = i
        self.features = np.ndarray((features.shape[0], features.shape[1]), dtype=np.float)
        for i in range(self.images.shape[0]):
            self.features[i] = features[img2idx[os.path.abspath(self.images[i])]]
        return self.features
        

    def merge_with(self, dataset, mix_labels=True):
        images = np.concatenate([self.images, dataset.images], axis=0)
        if mix_labels:
            labels = np.concatenate([self.labels, dataset.labels], axis=0)
        else:
            _, labels1 = np.unique(self.labels, return_inverse=True)
            _, labels2 = np.unique(dataset.labels, return_inverse=True)
            labels2 = labels2 + np.max(labels1)
            labels = np.concatenate([labels1, labels2], axis=0)
        if self.features is not None and dataset.features is not None:
            features = np.concatenate([self.features, dataset.features])
    
        new_dataset = type(self)()
        new_dataset.images = images
        new_dataset.labels = labels
        new_dataset.features = features
        new_dataset.init_classes()
        
        print('built new dataset: %d images of %d classes' % (len(new_dataset.images), new_dataset.num_classes))

        return new_dataset

    def build_subset_from_classes(self, classes, new_labels=True):

        if type(classes[0]) is not self.DataClass:
            try:
                classes = self.classes[classes]
            except:
                raise TypeError('The classes argument should be either self.DataClass or indices!')

        images = []
        labels = []
        features = []
        for i, c in enumerate(classes):
            n = len(c.indices)
            images.extend(self.images[c.indices])
            labels.extend([i] * n)
            if self.features is not None:
                features.append(self.features[c.indices,:].copy())
        subset = type(self)()
        subset.images = np.array(images, dtype=np.object)
        subset.labels = np.array(labels, dtype=np.int32)
        if self.features is not None:
            subset.features = np.concatenate(features, axis=0)
        if new_labels:
            _, subset.labels = np.unique(subset.labels, return_inverse=True)
        subset.init_classes()

        print('built subset: %d images of %d classes' % (len(subset.images), subset.num_classes))
        return subset

    def build_subset_from_indices(self, indices, new_labels=True):
        subset = type(self)()
        subset.images = self.images[indices]
        subset.labels = self.labels[indices]
        if self.features is not None:
            subset.features = self.features[indices]
        if new_labels:
            _, subset.labels = np.unique(subset.labels, return_inverse=True)
        subset.init_classes()

        print('built subset: %d images of %d classes' % (len(subset.images), subset.num_classes))
        return subset

    def separate_by_ratio(self, ratio, random_sort=True):
        num_classes = int(len(self.classes) * ratio)
        if random_sort:
            indices = np.random.permutation(len(self.classes))
        else:
            indices = np.arange(len(self.classes))
        indices1, indices2 = (indices[:num_classes], indices[num_classes:])
        classes1 = self.classes[indices1]
        classes2 = self.classes[indices2]
        return self.build_subset_from_classes(classes1), self.build_subset_from_classes(classes2)

    def split_k_folds(self, k, random_sort=True):
        self.k_folds_classes = []
        length = int(np.ceil(float(len(self.classes)) / k))
        if random_sort:
            indices = np.random.permutation(len(self.classes))
        else:
            indices = np.arange(len(self.classes))
        for i in range(k):
            start_ = i * length
            end_ = min(len(self.classes), (i+1) * length)
            self.k_folds_classes.append(self.classes[indices[start_:end_]])

    def get_fold(self, fold):
        k = len(self.k_folds_classes)
        assert fold <= k
        # Concatenate the classes in difference folds for trainset
        trainset_classes = [c for i in range(k) if i!=fold for c in self.k_folds_classes[i]]
        testset_classes = self.k_folds_classes[fold]
        trainset = self.build_subset_from_classes(trainset_classes)
        testset = self.build_subset_from_classes(testset_classes)
        return trainset, testset

    # Data Loading
    def init_index_queue(self, batch_format):
        if self.index_queue is None:
            self.index_queue = Queue()
        
        if batch_format in ['random_samples'] \
            or batch_format.startswith('random_samples_with_mates'):
            size = self.images.shape[0]
            index_queue = np.random.permutation(size)[:,None]
        elif batch_format in ['ordered_samples']:
            index_queue = np.arange(self.images.shape[0])[:,None]
        else:
            raise ValueError('IndexQueue: Unknown batch_format: {}!'.format(batch_format))
        for idx in list(index_queue):
            self.index_queue.put(idx)


    def get_batch(self, batch_size, batch_format, indices_only=False):
        ''' Get the indices from index queue and fetch the data with indices.'''
        indices_batch = []
        
        if batch_format =='random_samples' or batch_format=='ordered_samples':
            while len(indices_batch) < batch_size:
                indices_batch.extend(self.index_queue.get(block=True, timeout=30)) 
            assert len(indices_batch) == batch_size

        elif batch_format == 'random_pairs':
            assert batch_size%2 == 0
            classes = np.random.permutation(self.classes)[:batch_size//2]
            indices_batch = np.concatenate([c.random_pair() for c in classes], axis=0)

        elif batch_format.startswith('random_classes'):
            try:
                _, num_classes = batch_format.split(':')
                num_classes = int(num_classes)
            except:
                print('Use batch_format in such a format: random_classes: $NUM_CLASSES')
            assert batch_size % num_classes == 0
            num_samples_per_class = batch_size // num_classes
            idx_classes = np.random.permutation(self.num_classes)[:num_classes]
            indices_batch = []
            for data_class in self.classes[idx_classes]:
                indices_batch.extend(data_class.random_samples(num_samples_per_class))

        elif batch_format.startswith('random_samples_with_mates'):
            try:
                _, num_classes = batch_format.split(':')
                num_classes = int(num_classes)
            except:
                print('Use batch_format in such a format: random_classes: $NUM_CLASSES')
            num_samples_per_class = batch_size // num_classes
            assert batch_size % num_classes == 0
            while len(indices_batch) < batch_size:
                seed_idx = self.index_queue.get(block=True, timeout=30)
                assert len(seed_idx) == 1
                seed_idx = seed_idx[0]
                seed_class= self.idx2cls[seed_idx]
                # Make sure self.classes is in the order of labels
                assert seed_class.label == self.labels[seed_idx]
                indices_batch.extend([seed_idx] + \
                    list(seed_class.random_samples(num_samples_per_class-1, exception=seed_idx)))
            assert len(indices_batch) == batch_size
            
        else:
            raise ValueError('get_batch: Unknown batch_format: {}!'.format(batch_format))

        if indices_only:
            return indices_batch
    
        image_batch = self.images[indices_batch]
        label_batch = self.labels[indices_batch]
        batch = {
            'images': self.images[indices_batch],
            'labels': self.labels[indices_batch],
            'targets': self.targets[indices_batch]
        }
        return batch

    # Multithreading preprocessing images
    def start_index_queue(self, batch_format):
        if not (batch_format in ['random_samples', 'ordered_samples'] or \
            batch_format.startswith('random_samples_with_mates')):
            return
        self.index_queue = Queue()
        def index_queue_worker():
            while True:
                if self.index_queue.empty():
                    self.init_index_queue(batch_format)
                time.sleep(0.01)
        self.index_worker = Process(target=index_queue_worker)
        self.index_worker.daemon = True
        self.index_worker.start()

    def start_batch_queue(self, batch_size, batch_format, proc_func=None, maxsize=1, num_threads=3):
        if self.index_queue is None:
            self.start_index_queue(batch_format)

        self.batch_queue = Queue(maxsize=maxsize)
        def batch_queue_worker(seed):
            np.random.seed(seed)
            while True:
                batch = self.get_batch(batch_size, batch_format)
                if proc_func is not None:
                    batch['image_paths'] = batch['images']
                    batch['images'] = proc_func(batch['image_paths'])
                    batch['targets'] = proc_func(batch['targets'])
                self.batch_queue.put(batch)

        self.batch_workers = []
        for i in range(num_threads):
            worker = Process(target=batch_queue_worker, args=(i,))
            worker.daemon = True
            worker.start()
            self.batch_workers.append(worker)
    
    def pop_batch_queue(self, timeout=60):
        return self.batch_queue.get(block=True, timeout=timeout)
      
    def release_queue(self):
        if self.index_queue is not None:
            self.index_queue.close()
        if self.batch_queue is not None:
            self.batch_queue.close()
        if self.index_worker is not None:
            self.index_worker.terminate()   
            del self.index_worker
            self.index_worker = None
        if self.batch_workers is not None:
            for w in self.batch_workers:
                w.terminate()
                del w
            self.batch_workers = None

