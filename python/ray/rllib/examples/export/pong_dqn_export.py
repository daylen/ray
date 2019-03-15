#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import ray
import tensorflow as tf
import numpy as np

from ray.rllib.agents.registry import get_agent_class

ray.init(num_cpus=4)

"""
Train a model. Pass in a dictionary for the transfer_weights param to do transfer learning.
"""
def train_and_export(algo_name, num_steps, ckpt_dir, prefix, transfer_weights=None):
    cls = get_agent_class(algo_name)
    alg = cls(config={
        "double_q": True,
        "dueling": True,
        "num_atoms": 1,
        "noisy": False,
        "prioritized_replay": False,
        "n_step": 1,
        "target_network_update_freq": 8000,
        "gamma": 0.99,
        "lr": .0001,
        "adam_epsilon": .00015,
        "hiddens": [512],
        "learning_starts": 10000,
        "buffer_size": 50000,
        "sample_batch_size": 4,
        "train_batch_size": 32,
        "schedule_max_timesteps": 2000000,
        "exploration_final_eps": 0.01,
        "exploration_fraction": .1,
        "prioritized_replay_alpha": 0.5,
        "beta_annealing_fraction": 1.0,
        "final_prioritized_replay_beta": 1.0,
        "num_gpus": 1,
        "timesteps_per_iteration": 10000,
    }, env="PongDeterministic-v4")

    # Set transfer weights if we have them.
    if transfer_weights is not None:
        print('Setting transfer weights for keys:', transfer_weights.keys());
        alg.get_policy().set_weights_dict(transfer_weights)

    for i in range(num_steps):
        print('Training iter', i)
        alg.train()
        # Export tensorflow checkpoint for fine-tuning
        if (i%3 == 0):
            alg.export_policy_checkpoint(ckpt_dir, filename_prefix=prefix + "_" + str(i) + ".ckpt")
    alg.export_policy_checkpoint(ckpt_dir, filename_prefix=prefix + "_FINAL.ckpt")

"""
Return dictionary mapping var names to weights.
"""
def read_checkpoint_conv_only(export_dir, prefix):
    sess = tf.Session()
    meta_file = "%s.meta" % prefix
    saver = tf.train.import_meta_graph(os.path.join(export_dir, meta_file))
    saver.restore(sess, os.path.join(export_dir, prefix))

    # clear weights
    init = tf.global_variables_initializer()
    sess.run(init)

    # Type: array of tensorflow vars
    tf_vars_to_restore = [v for v in tf.trainable_variables() if "conv" in v.name]
    # Type: array of strings
    var_names_to_restore = [v.name for v in tf_vars_to_restore]

    # Overwrite saver
    saver = tf.train.Saver(tf_vars_to_restore)
    saver.restore(sess, os.path.join(export_dir, prefix))
    weights_dict = {}
    for v in tf.trainable_variables():
        value = sess.run(v)
        print(v.name, np.mean(value))
        if v.name in var_names_to_restore:
            # We need to chop off the :0 part of the var name
            weights_dict[v.name.replace(':0', '')] = value
    return weights_dict


if __name__ == "__main__":
    algo = "DQN"
    ckpt_dir = "/tmp/ckpt_export_dir"
    if not os.path.isdir(ckpt_dir):
        print("Directory does not exist!" + 5 / 0)
    num_steps = 1
    # print('BEGIN TRAIN ONE STEP')
    # train_and_export(algo, num_steps, ckpt_dir, "model")
    print('BEGIN LOAD VARS')
    weights = read_checkpoint_conv_only(ckpt_dir, "model_FINAL.ckpt")
    print('BEGIN RETRAIN')
    train_and_export(algo, 1000000, ckpt_dir, "transfer_model", weights)
