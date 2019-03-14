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


def train_and_export(algo_name, num_steps, model_dir, ckpt_dir, prefix):
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

    for i in range(num_steps):
        if i % 2000 == 0:
            print('Training iter', i)
        alg.train()
        if i % 10000 == 0:
            # Export tensorflow checkpoint for fine-tuning
            alg.export_policy_checkpoint(ckpt_dir, filename_prefix=prefix + "_" + str(i) + ".ckpt")
            # Export tensorflow SavedModel for online serving
            # alg.export_policy_model(model_dir)
    alg.export_policy_checkpoint(ckpt_dir, filename_prefix=prefix + "_FINAL.ckpt")


def restore_saved_model(export_dir):
    signature_key = \
        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    g = tf.Graph()
    with g.as_default():
        with tf.Session(graph=g) as sess:
            meta_graph_def = \
                tf.saved_model.load(sess,
                                    [tf.saved_model.tag_constants.SERVING],
                                    export_dir)
            print("Model restored!")
            print("Signature Def Information:")
            print(meta_graph_def.signature_def[signature_key])
            print("You can inspect the model using TensorFlow SavedModel CLI.")
            print("https://www.tensorflow.org/guide/saved_model")


def restore_checkpoint(export_dir, prefix):
    sess = tf.Session()
    meta_file = "%s.meta" % prefix
    saver = tf.train.import_meta_graph(os.path.join(export_dir, meta_file))
    saver.restore(sess, os.path.join(export_dir, prefix))
    print("Checkpoint restored!")
    print("Variables Information:")
    for v in tf.trainable_variables():
        #value = sess.run(v)
        #print(v.name, value)
        print(v.name)

def restore_checkpoint_conv_only(export_dir, prefix):
    sess = tf.Session()
    meta_file = "%s.meta" % prefix
    saver = tf.train.import_meta_graph(os.path.join(export_dir, meta_file))
    saver.restore(sess, os.path.join(export_dir, prefix))

    # clear weights
    init = tf.global_variables_initializer()
    sess.run(init)

    vars_to_restore = [v for v in tf.trainable_variables() if "conv" in v.name]
    print(vars_to_restore)

    # new saver
    saver = tf.train.Saver(vars_to_restore)
    saver.restore(sess, os.path.join(export_dir, prefix))
    for v in tf.trainable_variables():
        value = sess.run(v)
        print(v.name, np.mean(value))


if __name__ == "__main__":
    algo = "DQN"
    model_dir = ""
    ckpt_dir = "/results/ckpt_export_dir"
    if not os.path.isdir(ckpt_dir):
        print("Oh no" + 5 / 0)
    prefix = "model"
    num_steps = 5000000
    train_and_export(algo, num_steps, model_dir, ckpt_dir, prefix)
    # restore_saved_model(model_dir)
    # restore_checkpoint_conv_only(ckpt_dir, prefix)
