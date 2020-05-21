import os
import sys
import time
import argparse
import tensorflow as tf
import numpy as np
from functools import partial

from utils import utils
from utils.dataset import Dataset
from utils.imageprocessing import preprocess, flip

from advfaces import AdvFaces
from base_network import BaseNetwork

###########################################################
### Generate adversarial images and masks for visualization
###########################################################
def test(network, config, original_images, targets, log_dir, step):
    output_dir = os.path.join(log_dir, "samples")
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    if config.mode == "target":
        generated, r = network.generate_images(original_images, targets)
    else:
        generated, r = network.generate_images(original_images, targets=None)
    utils.save_manifold(generated, os.path.join(output_dir, "{}_gen.jpg".format(step)))
    utils.save_manifold(r, os.path.join(output_dir, "{}_r.jpg".format(step)))

###########################################################
### Compute Attack Success Rate                         ###
###########################################################
def success_rate(
    network,
    config,
    original_images,
    targets,
    target_feats,
    log_dir,
    step,
):
    if config.mode == "target":
        fakes, _ = network.generate_images(original_images, targets)
    else:
        fakes, _ = network.generate_images(original_images, targets=None)
    gen_feats = network.aux_matcher_extract_feature(fakes, batch_size=512, verbose=True)

    scores_a_t = utils.cosine_pair(gen_feats, target_feats)
    if config.mode == 'target':
        sr = (sum(scores_a_t > config.aux_matcher_threshold) / len(scores_a_t)) * 100
    else:
        sr = (sum(scores_a_t <= config.aux_matcher_threshold) / len(scores_a_t)) * 100
    print("Success Rate: {}%".format(sr))
    print("Mean Sim. Score (adv v. target): {}", format(np.mean(scores_a_t)))
    with open(log_dir + "/accuracy.txt", "a") as f:
        f.write("{}: {}\n".format(sr, step))
    return sr, np.mean(scores_a_t)

def main(args):
    config_file = args.config_file
    # I/O
    config = utils.import_file(config_file, "config")

    trainset = Dataset(config.train_dataset_path, config.mode)
    testset = Dataset(config.test_dataset_path, config.mode)

    network = AdvFaces()
    network.initialize(config, trainset.num_classes)
    

    # Initalization for running
    log_dir = utils.create_log_dir(config, config_file)
    summary_writer = tf.summary.FileWriter(log_dir, network.graph)

    if config.restore_model:
        network.restore_model(config.restore_model, config.restore_scopes)
    proc_func = lambda images: preprocess(images, config, True)
    trainset.start_batch_queue(
        config.batch_size, batch_format=config.batch_format, proc_func=proc_func
    )

    #
    # Main Loop
    #
    print(
        "\nStart Training\n# epochs: %d\nepoch_size: %d\nbatch_size: %d\n"
        % (config.num_epochs, config.epoch_size, config.batch_size)
    )
    global_step = 0
    start_time = time.time()
    for epoch in range(config.num_epochs):

        if epoch == 0:
            print("Loading Test Set")
            originals = preprocess(
                testset.images, config, is_training=False
            )
            targets = preprocess(testset.targets, config, False)
            print('Done loading test set')
            test_images = np.squeeze(originals[np.where(testset.labels < 5)[0]])
            target_feats = network.aux_matcher_extract_feature(targets)
            output_dir = os.path.join(log_dir, "samples")
            if not os.path.isdir(output_dir):
                os.makedirs(output_dir)
            utils.save_manifold(test_images, os.path.join(output_dir, "original.jpg"))
            print("Computing initial success rates..")
            success_rate(
                network,
                config,
                originals,
                targets,
                target_feats,
                log_dir,
                global_step
            )
            print("testing.")
            test(
                network,
                config,
                test_images,
                targets,
                log_dir,
                global_step,
            )

        # Training
        for step in range(config.epoch_size):
            # Prepare input
            learning_rate = utils.get_updated_learning_rate(global_step, config)
            batch = trainset.pop_batch_queue()
            wl, sm, global_step = network.train(
                batch["images"],
                batch["targets"],
                batch["labels"],
                learning_rate,
                config.keep_prob,
                trainset.num_classes,
            )
            wl["lr"] = learning_rate

            # Display
            if step % config.summary_interval == 0:
                duration = time.time() - start_time
                start_time = time.time()
                utils.display_info(epoch, step, duration, wl)
                summary_writer.add_summary(sm, global_step=global_step)

        # Computing success rate
        success_rate(
                network,
                config,
                originals,
                targets,
                target_feats,
                log_dir,
                global_step
        )

        # Testing
        test(
            network,
            config,
            test_images,
            targets,
            log_dir,
            global_step,
        )

        # Save the model
        network.save_model(log_dir, global_step)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_file", help="The path to the training configuration file", type=str
    )
    args = parser.parse_args()
    main(args)
