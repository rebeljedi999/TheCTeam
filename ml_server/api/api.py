from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import flask
from flask import request, jsonify


import numpy as np
import tensorflow as tf
from tf_agents.agents import tf_agent

from tf_agents.agents.reinforce import reinforce_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import actor_distribution_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
# from .environment import AgentEnv
from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.trajectories import time_step as ts
from tf_agents.networks import network
from tf_agents.specs import array_spec
from tf_agents.specs import tensor_spec
from tf_agents.policies import actor_policy

tf.compat.v1.enable_v2_behavior()
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
"""
Policy
"""


class ActionNet(network.Network):

    def __init__(self, input_tensor_spec, output_tensor_spec):
        super(ActionNet, self).__init__(
            input_tensor_spec=input_tensor_spec,
            state_spec=(),
            name='ActionNet')
        self._output_tensor_spec = output_tensor_spec
        self._sub_layers = [
            tf.keras.layers.Dense(
                8, activation=tf.nn.softmax),
        ]

    def call(self, observations, step_type, network_state):
        del step_type

        output = tf.cast(observations, dtype=tf.float32)
        for layer in self._sub_layers:
            output = layer(output)
        output = np.argmax(output)
        print(output)
        actions = tf.reshape(
            output, [-1] + self._output_tensor_spec.shape.as_list())

        # Scale and shift actions to the correct range if necessary.
        return actions, network_state


input_tensor_spec = tensor_spec.TensorSpec((1, 91), tf.float32)
#print('Input tensor spec: ')
#print(input_tensor_spec)

time_step_spec = ts.time_step_spec(input_tensor_spec)
#print('Time step spec: ')
#print(time_step_spec)

action_spec = tensor_spec.BoundedTensorSpec(
    (), np.int64, minimum=0, maximum=8)
#print('-')

action_net = ActionNet(input_tensor_spec, action_spec)


my_actor_policy = actor_policy.ActorPolicy(
    time_step_spec=time_step_spec,
    action_spec=action_spec,
    actor_network=action_net)

"""
Troubleshooting statements
Checking shapes, etc
"""
#observation = tf.ones([1] + time_step_spec.observation.shape.as_list())
#print(time_step_spec.observation.shape.as_list())
#observation = [0, 5, 6, 3]
#observation = tf.constant([0, 5, 6, 3], shape=(1, 4), dtype=tf.float32)
#print('Observation: ')
#print(observation)
#time_step = ts.restart(observation)
#print('Time step: ')
#print(time_step)

#action_step = my_actor_policy.action(time_step)
#print('Action: ')
#print(action_step)
"""
Etc
"""
class Agent(tf_agent.TFAgent):
  def __init__(self):
    self._situation = tf.compat.v2.Variable(0, dtype=tf.int32)
    policy = my_actor_policy
    time_step_spec = policy.time_step_spec
    action_spec = policy.action_spec
    super(Agent, self).__init__(time_step_spec=time_step_spec, action_spec=action_spec, policy=policy, collect_policy=policy, train_sequence_length=None)

  def _initialize(self):
    return tf.compat.v1.variables_initializer(self.variables)

  def _train(self, experience, weights=None):
    observation = experience.observation
    action = experience.action
    reward = experience.reward
    return tf_agent.LossInfo((), ())

agent = Agent()


"""
Turns an initial observation, the resulting action, and the observation after that action
into a trajectory, which can be used to train the model.
"""

def trajectory_for_training(initial_step, action_step, final_step):
    return trajectory.Trajectory(observation=tf.expand_dims(initial_step.observation, 0),
                                 action=tf.expand_dims(action_step.action, 0),
                                 policy_info=action_step.info,
                                 reward=tf.expand_dims(final_step.reward, 0),
                                 discount=tf.expand_dims(
                                     final_step.discount, 0),
                                 step_type=tf.expand_dims(
                                     initial_step.step_type, 0),
                                 next_step_type=tf.expand_dims(final_step.step_type, 0))


local_data = {}

###############################################################
app = flask.Flask(__name__)
app.config["DEBUG"] = True

@app.route('/start', methods=['POST'])
def observe():
    observation = request.get_json(force=True)
    print(observation)
    obs = tf.constant(observation["visualSensor"], shape=(1,91), dtype=tf.float32)
    time_step = ts.restart(obs)
    action_step = my_actor_policy.action(time_step)
    action = action_step.action.numpy()[0]
    local_data["action"] = action_step
    local_data["step"] = time_step
    return jsonify({'action': float(action)})



@app.route('/train', methods=['POST'])
def train():
    training = request.get_json(force=True)
    reward = training["reward"]
    obs = tf.constant(training["visualSensor"], shape =(1,91), dtype=tf.float32)
    time_step = ts.transition(obs, reward)
    last_step = local_data["step"]
    action = local_data["action"]
    experience = trajectory_for_training(last_step, action, time_step)
    agent._train(experience)
    local_data["step"] = time_step
    time_step = ts.transition(obs, reward)
    action_step = my_actor_policy.action(time_step)
    action = action_step.action.numpy()[0]
    local_data["action"] = action_step
    return jsonify({'action': float(action)})


@app.route('/testing', methods=['POST'])
def test():
    print(request.get_json(Force=True))


app.run()