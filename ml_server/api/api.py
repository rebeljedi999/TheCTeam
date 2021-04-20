from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import flask
from flask import request, jsonify
import numpy as np
import tensorflow as tf

from tf_agents.agents import tf_agent
from tf_agents.networks import actor_distribution_network
from tf_agents.agents.reinforce import reinforce_agent
from tf_agents.bandits.metrics import tf_metrics
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.networks import network
from tf_agents.specs import array_spec
from tf_agents.specs import tensor_spec
from tf_agents.policies import actor_policy
import pandas as pd
import os
import pathlib
from datetime import datetime

tf.compat.v1.enable_v2_behavior()

# activate GPU if one is available
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


"""
Shape specifications
"""
input_tensor_spec = tensor_spec.TensorSpec((91), tf.float32)
time_step_spec = ts.time_step_spec(input_tensor_spec)
action_spec = tensor_spec.BoundedTensorSpec((), tf.int64, minimum=0, maximum=8)


"""
Agent
"""
actor_net = actor_distribution_network.ActorDistributionNetwork(
    input_tensor_spec,
    action_spec,
    fc_layer_params=(100,))

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001, epsilon=1e-8)

train_step_counter = tf.compat.v2.Variable(0)

agent = reinforce_agent.ReinforceAgent(
    time_step_spec,
    action_spec,
    actor_network=actor_net,
    optimizer=optimizer,
    normalize_returns=True,
    train_step_counter=train_step_counter)

agent.initialize()

"""
Turns an initial observation, the resulting action, and the observation after that action
into a trajectory, which can be used to train the model.
"""

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=1,
    max_length=120)


def trajectory_creator(initial_step, action_step, final_step):
  return trajectory.Trajectory(observation=tf.expand_dims(initial_step.observation, 0),
                               action=tf.expand_dims(action_step.action, 0),
                               policy_info=action_step.info,
                               reward=tf.expand_dims(final_step.reward, 0),
                               discount=tf.expand_dims(final_step.discount, 0),
                               step_type=tf.expand_dims(initial_step.step_type, 0),
                               next_step_type=tf.expand_dims(final_step.step_type, 0))


local_data = {}
local_data["buffer"] = replay_buffer
###############################################################
app = flask.Flask(__name__)
app.config["DEBUG"] = True

@app.route('/start', methods=['POST'])
def observe():
    observation = request.get_json(force=True)
    print(observation)
    obs = tf.constant(observation["visualSensor"], shape=(91), dtype=tf.float32)
    time_step = ts.restart(obs)
    action_step = agent.collect_policy.action(time_step)
    local_data["action"] = action_step
    local_data["step"] = time_step
    action = action_step.action.numpy()
    return jsonify({'action': float(action)})


@app.route('/step', methods=['POST'])
def step():
    training = request.get_json(force=True)
    reward = training["reward"]
    obs = tf.constant(training["visualSensor"],
                      shape=(91), dtype=tf.float32)
    time_step = ts.transition(obs, reward)
    last_step = local_data["step"]
    action = local_data["action"]
    experience = trajectory_creator(last_step, action, time_step)
    print("Buffer:" )
    print(local_data["buffer"])
    print("Experience:" )
    print(experience)
    local_data["buffer"].add_batch(experience)
    local_data["step"] = time_step
    time_step = ts.transition(obs, reward)
    action_step = agent.collect_policy.action(time_step)
    action = action_step.action.numpy()
    local_data["action"] = action_step
    return jsonify({'action': float(action)})


@app.route('/train', methods=['POST'])
def train():
    training = request.get_json(force=True)
    reward = training["reward"]
    obs = tf.constant(training["visualSensor"],
                      shape=(1, 91), dtype=tf.float32)
    time_step = ts.termination(obs, reward)
    last_step = local_data["step"]
    action = local_data["action"]
    experience = trajectory_creator(last_step, action, time_step)
    local_data["buffer"].add_batch(experience)
    data = local_data["buffer"].gather_all()
    tl = agent._train(data)
    local_data["buffer"].clear()
    return jsonify({"Result": "Success"})

@app.route('/analyze', methods=['POST'])
def analyze():
    # get data and save to csv
    data = request.get_json(force=True)
    df = pd.DataFrame(dict(data))
    now = datetime.now()
    file_name = str(pathlib.Path(__file__).parent.absolute()) + "/data_readout/data-" +\
                now.strftime("%m-%d-%Y_%H-%M-%S") + ".csv"
    df.to_csv(file_name, index=False)
    print("data saved")
    return jsonify({"Result": "Success"})

app.run()