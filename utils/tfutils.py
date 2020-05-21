import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim




###############################################################################
############################ TENSOR EVALUATION ################################
###############################################################################
def rank_accuracy(logits, label, k=1):
    batch_size = tf.shape(logits)[0]
    _, arg_top = tf.nn.top_k(logits, k)
    label = tf.cast(label, tf.int32)
    label = tf.reshape(label, [batch_size, 1])
    label = tf.tile(label, [1, k])
    correct = tf.reduce_any(tf.equal(label, arg_top), axis=1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    return accuracy

def cosine_pair(x1, x2):
    #assert x1.shape == x2.shape
    eps = 1e-10
    x1_norm = tf.sqrt(tf.reduce_sum(tf.square(x1), axis=1, keepdims=True))
    x2_norm = tf.sqrt(tf.reduce_sum(tf.square(x2), axis=1, keepdims=True))
    x1 = x1 / (x1_norm + eps)
    x2 = x2 / (x2_norm + eps)
    dist = tf.reduce_sum(x1 * x2, axis=1)
    return dist

def convert_to_classifier(x1, x2):
    distance = tf.reduce_sum(tf.square(x1 - x2), axis=1)
    threshold = 0.99
    score = tf.where(distance > threshold,
                    0.5 + ((distance - threshold) * 0.5) / (4.0 - threshold),
                    0.5 * distance / threshold)
    reverse_score = 1.0 - score
    return tf.transpose(tf.stack([reverse_score, score]))

def cosine_pair_np(x1, x2):
    assert x1.shape == x2.shape
    epsilon = 1e-10
    x1_norm = np.sqrt(np.sum(np.square(x1), axis=1, keepdims=True))
    x2_norm = np.sqrt(np.sum(np.square(x2), axis=1, keepdims=True))
    x1 = x1 / (x1_norm+epsilon)
    x2 = x2 / (x2_norm+epsilon)
    dist = np.sum(x1 * x2, axis=1)
    return dist

###############################################################################
###################### TENSORFLOW RELATED OPS #################################
###############################################################################
def average_tensors(tensors, name=None):
    if len(tensors) == 1:
        return tf.identity(tensors[0], name=name)
    else:
        # Each tensor in the list should be of the same size
        expanded_tensors = []

        for t in tensors:
            expanded_t = tf.expand_dims(t, 0)
            expanded_tensors.append(expanded_t)

        average_tensor = tf.concat(axis=0, values=expanded_tensors)
        average_tensor = tf.reduce_mean(average_tensor, 0, name=name)

        return average_tensor


def average_grads(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
        tower_grads: List of lists of gradients. The outer list is over different 
        towers. The inner list is over the gradient calculation in each tower.
    Returns:
        List of gradients where the gradient has been averaged across all towers.
    """
    if len(tower_grads) == 1:
        return tower_grads[0]
    else:
        average_grads = []
        for grad_ in zip(*tower_grads):
            # Note that each grad looks like the following:
            #   (grad0_gpu0, ... , grad0_gpuN)
            average_grad = None if grad_[0] == None else average_tensors(grad_)
            average_grads.append(average_grad)

        return average_grads


def apply_gradient(
    update_gradient_vars,
    grads,
    optimizer,
    learning_rate,
    learning_rate_multipliers=None,
):
    assert len(grads) == len(update_gradient_vars)
    if learning_rate_multipliers is None:
        learning_rate_multipliers = {}
    # Build a dictionary to save multiplier config
    # format -> {scope_name: ((grads, vars), lr_multi)}
    learning_rate_dict = {}
    learning_rate_dict["__default__"] = ([], 1.0)
    for scope, multiplier in learning_rate_multipliers.items():
        assert scope != "__default__"
        learning_rate_dict[scope] = ([], multiplier)

    # Scan all the variables, insert into dict
    scopes = learning_rate_dict.keys()
    for var, grad in zip(update_gradient_vars, grads):
        count = 0
        scope_temp = ""
        for scope in scopes:
            if scope in var.name:
                scope_temp = scope
                count += 1
        assert count <= 1, (
            "More than one multiplier scopes appear in variable: %s" % var.name
        )
        if count == 0:
            scope_temp = "__default__"
        if grad is not None:
            learning_rate_dict[scope_temp][0].append((grad, var))

    # Build a optimizer for each multiplier scope
    apply_gradient_ops = []
    print("\nLearning rate multipliers:")
    for scope, scope_content in learning_rate_dict.items():
        scope_grads_vars, multiplier = scope_content
        if type(multiplier) is tuple:
            optimizer_name, optimizer_params, scope_multiplier = multiplier
        else:
            optimizer_name, optimizer_params = optimizer
            scope_multiplier = multiplier
        scope_learning_rate = scope_multiplier * learning_rate

        skipped = len(scope_grads_vars) == 0 or scope_multiplier == 0
        warning = "\033[92m (SKIPPED)\033[0m" if skipped else ""
        print(
            "{}{}:\n--#variables: {}\n--optimizer:{}\n--lr_multi: {}".format(
                scope,
                warning,
                len(scope_grads_vars),
                (optimizer_name, optimizer_params),
                scope_multiplier,
            )
        )
        if skipped:
            continue
        if optimizer_name == "ADAGRAD":
            opt = tf.train.AdagradOptimizer(scope_learning_rate, **optimizer_params)
        elif optimizer_name == "ADADELTA":
            opt = tf.train.AdadeltaOptimizer(scope_learning_rate, **optimizer_params)
        elif optimizer_name == "ADAM":
            opt = tf.train.AdamOptimizer(scope_learning_rate, **optimizer_params)
        elif optimizer_name == "RMSPROP":
            opt = tf.train.RMSPropOptimizer(scope_learning_rate, **optimizer_params)
        elif optimizer_name == "MOM":
            opt = tf.train.MomentumOptimizer(scope_learning_rate, **optimizer_params)
        elif optimizer_name == "SGD":
            opt = tf.train.GradientDescentOptimizer(scope_learning_rate)
        else:
            raise ValueError("Invalid optimization algorithm")
        apply_gradient_ops.append(opt.apply_gradients(scope_grads_vars))
    print("")
    apply_gradient_op = tf.group(*apply_gradient_ops)

    return apply_gradient_op


def save_model(sess, saver, model_dir, global_step):
    with sess.graph.as_default():
        checkpoint_path = os.path.join(model_dir, "ckpt")
        metagraph_path = os.path.join(model_dir, "graph.meta")

        print("Saving variables...")
        saver.save(
            sess, checkpoint_path, global_step=global_step, write_meta_graph=False
        )
        if not os.path.exists(metagraph_path):
            print("Saving metagraph...")
            saver.export_meta_graph(metagraph_path)


def restore_model(sess, var_list, model_dir, restore_scopes=None, replace=None):
    """ Load the variable values from a checkpoint file into pre-defined graph.
    Filter the variables so that they contain at least one of the given keywords."""
    with sess.graph.as_default():
        if restore_scopes is not None:
            var_list = [
                var
                for var in var_list
                if any([scope in var.name for scope in restore_scopes])
            ]
        if replace is not None:
            var_dict = {}
            for var in var_list:
                name_new = var.name
                for k, v in replace.items():
                    name_new = name_new.replace(k, v)
                name_new = name_new[:-2]  # When using dict, numbers should be removed
                var_dict[name_new] = var
            var_list = var_dict
        model_dir = os.path.expanduser(model_dir)
        ckpt_file = tf.train.latest_checkpoint(model_dir)

        print("Restoring {} variables from {} ...".format(len(var_list), ckpt_file))
        saver = tf.train.Saver(var_list)
        saver.restore(sess, ckpt_file)


def load_model(sess, model_path, scope=None):
    """ Load the the graph and variables values from a model path.
    Model path is either a a frozen graph or a directory with both
    a .meta file and checkpoint files."""
    print("LOADING MODEL FROM {}".format(model_path))
    with sess.graph.as_default():
        model_path = os.path.expanduser(model_path)
        if os.path.isfile(model_path):
            # Frozen grpah
            print("Model filename: %s" % model_path)
            with gfile.FastGFile(model_path, "rb") as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name="")
        else:
            # Load grapha and variables separatedly.
            meta_files = [
                file for file in os.listdir(model_path) if file.endswith(".meta")
            ]
            assert len(meta_files) == 1
            meta_file = os.path.join(model_path, meta_files[0])
            ckpt_file = tf.train.latest_checkpoint(model_path)

            print("Metagraph file: %s" % meta_file)
            print("Checkpoint file: %s" % ckpt_file)
            saver = tf.train.import_meta_graph(
                meta_file, clear_devices=True, import_scope=scope
            )
            saver.restore(sess, ckpt_file)


###############################################################################
############################ TRAINING MONITOR #################################
###############################################################################
default_list = 'main'
watchlists = {}
conflict_solution = "latest" # solution when key conflicts, one of followings ["oldest", "latest", "exception"]

def set_default(listname):
	default_list = listname

def list_all_lists():
	return watchlists.keys()


def insert(key, var, listname=None):
	# Get wathclist, if not exist, create it
	if listname is None:
		listname = default_list
	if not listname in watchlists:
		watchlists[listname] = {}
	watchlist = watchlists[listname]

	# Insert the variable into the list
	assert type(key) is str, "the key for tfwatcher must be an string!"
	if key in watchlist.keys():
		if conflict_solution == "oldest":
			pass
		elif conflict_solution == "latest":
			watchlist[key] = var
		elif conflict_solution == "exception":
			raise KeyError("Trying to insert an variable with key {} \
				when it is already in the watchlist of tfwatcher!".format(key))
		else:
			raise ValueError("Unknown conflict_solution in tfwatcher: {}".format(conflict_solution))
	else:
		watchlist[key] = var

def get_watchlist(listname=None):
	if listname is None:
		listname = default_list
	return watchlists[listname]