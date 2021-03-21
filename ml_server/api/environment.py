from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.networks import network
from tf_agents.policies import tf_policy
from tf_agents.specs import array_spec
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory

from tf_agents.policies import actor_policy
from tf_agents.networks import actor_distribution_network
from tf_agents.agents.reinforce import reinforce_agent
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common

import tensorflow as tf
import numpy as np

class AgentEnv(py_environment.PyEnvironment):

  def __init__(self):
    self._action_spec = array_spec.BoundedArraySpec(
        shape=(), dtype=np.int32, minimum=0, maximum=7, name='action')
    self._observation_spec = array_spec.BoundedArraySpec(
        shape=(1,), dtype=np.int32, minimum=0, name='observation')
    self._state = 0
    self._episode_ended = False

  def action_spec(self):
    return self._action_spec

  def observation_spec(self):
    return self._observation_spec

  def _reset(self):
    self._state = 0
    self._episode_ended = False
    return ts.restart(np.array([self._state], dtype=np.int32))

  def _step(self, action):

    if self._episode_ended:
      return self.reset()

    if action == 0:
        print(f"Action: {action}")
        self._episode_ended = True
      ## Send JSON
    elif action == 1:
        print(f"Action: {action}")
        ## 
    elif action == 2:
        print(f"Action: {action}")
        ##
    elif action == 3:
        print(f"Action: {action}")
        ##
    elif action == 4:
        print(f"Action: {action}")
        ##
    elif action == 5:
        print(f"Action: {action}")
        ##
    elif action == 6:
        print(f"Action: {action}")
        ##
    elif action == 7:
        print(f"Action: {action}")
        self._episode_ended = True

    else:
      raise ValueError('Invalid action.')

    if self._episode_ended:
      reward = self._state
      return ts.termination(np.array([self._state], dtype=np.int32), reward)
    else:
      return ts.transition(
          np.array([self._state], dtype=np.int32), reward=0.0, discount=1.0)


##Action network, as defined in documentation
class ActionNet(network.Network):

  def __init__(self, input_tensor_spec, output_tensor_spec):
    super(ActionNet, self).__init__(
        input_tensor_spec=input_tensor_spec,
        state_spec=(),
        name='ActionNet')
    self._output_tensor_spec = output_tensor_spec
    self._sub_layers = [
        tf.keras.layers.Dense(
            action_spec.shape.num_elements(), activation=tf.nn.tanh),
    ]

  def call(self, observations, step_type, network_state):
    del step_type

    output = tf.cast(observations, dtype=tf.float32)
    for layer in self._sub_layers:
      output = layer(output)
    actions = tf.reshape(output, [-1] + self._output_tensor_spec.shape.as_list())

    # Scale and shift actions to the correct range if necessary.
    return actions, network_state
##Testing environment
env = tf_py_environment.TFPyEnvironment(AgentEnv())
#utils.validate_py_environment(environment, episodes=5)
#######################
#######################
#TESTING AND INFO
#######################
#######################
print('Observation Spec:')
print(env.time_step_spec().observation)
print('Action Spec:')
print(env.action_spec())

time_step = env.reset()
print('Time step:')
print(time_step)

action = np.array(1, dtype=np.int32)

next_time_step = env.step(action)
print('Next time step:')
print(next_time_step)
#######################
#######################
# AGENT
#######################
#######################
#PARAMETERS
fc_layer_params = (100,)
learning_rate = 1e-3 # @param {type:"number"}
replay_buffer_capacity = 2000 # @param {type:"integer"}
log_interval = 25 # @param {type:"integer"}
num_eval_episodes = 10 # @param {type:"integer"}
num_iterations = 250
collect_episodes_per_iteration = 2
eval_interval = 50

#DEFINE ACTOR NETWORK
actor_net = actor_distribution_network.ActorDistributionNetwork(
    env.observation_spec(),
    env.action_spec(),
    fc_layer_params=fc_layer_params)
#SELECT OPTIMIZER
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

train_step_counter = tf.compat.v2.Variable(0)
#DEFINE AGENT
tf_agent = reinforce_agent.ReinforceAgent(
    env.time_step_spec(),
    env.action_spec(),
    actor_network=actor_net,
    optimizer=optimizer,
    normalize_returns=True,
    train_step_counter=train_step_counter)
#INITIALIZE AGENT
tf_agent.initialize()

eval_policy = tf_agent.policy
collect_policy = tf_agent.collect_policy

print("Collect data spec: ")
print(tf_agent.collect_data_spec)

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=tf_agent.collect_data_spec,
    batch_size=env.batch_size,
    max_length=replay_buffer_capacity)

def compute_avg_return(environment, policy, num_episodes=10):

  total_return = 0.0
  for _ in range(num_episodes):

    time_step = environment.reset()
    episode_return = 0.0

    while not time_step.is_last():
      action_step = policy.action(time_step)
      time_step = environment.step(action_step.action)
      episode_return += time_step.reward
    total_return += episode_return

  avg_return = total_return / num_episodes
  return avg_return.numpy()[0]



def collect_episode(environment, policy, num_episodes):

  episode_counter = 0
  environment.reset()

  while episode_counter < num_episodes:
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)

    replay_buffer.add_batch(traj)

    episode_counter += 1


###########
###########
# MAD
# SCIENTIST
###########
###########

# (Optional) Optimize by wrapping some of the code in a graph using TF function.
tf_agent.train = common.function(tf_agent.train)

# Reset the train step
tf_agent.train_step_counter.assign(0)

# Evaluate the agent's policy once before training.
avg_return = compute_avg_return(env, tf_agent.policy, num_eval_episodes)
returns = [avg_return]

for _ in range(num_iterations):

  # Collect a few episodes using collect_policy and save to the replay buffer.
  collect_episode(
      env, tf_agent.collect_policy, collect_episodes_per_iteration)

  # Use data from the buffer and update the agent's network.
  experience = replay_buffer.gather_all()
  print('Experience: ')
  print('-------------------------------------')
  print(experience)
  print('-------------------------------------')
  train_loss = tf_agent.train(experience)
  replay_buffer.clear()

  step = tf_agent.train_step_counter.numpy()

  if step % log_interval == 0:
    print('step = {0}: loss = {1}'.format(step, train_loss.loss))

  if step % eval_interval == 0:
    avg_return = compute_avg_return(env, tf_agent.policy, num_eval_episodes)
    print('step = {0}: Average Return = {1}'.format(step, avg_return))
    returns.append(avg_return)