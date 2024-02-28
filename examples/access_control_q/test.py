# Author: Seungjae Shin <sjshin0505@{etri.re.kr, gmail.com}>

"""Testing script for AccessControlQueue environment implemented based on the following reference:
R. S. Sutton and A. G. Barto, Reinforcement Learning - An Introduction, 2nd ed., 2018 (Example 10.2: An Access-Control
Queuing Task).
"""
"""
/home/labry/GymProxy/venv/bin/python /home/labry/GymProxy/examples/access_control_q/test.py 
[15:22:48 2024-02-23|INFO|main] 0-th step in 0-th episode / obs: (1, 9) / reward: 8.0 / done: False / info: {} / action: True
[15:22:48 2024-02-23|INFO|main] 1-th step in 0-th episode / obs: (1, 9) / reward: 1.0 / done: False / info: {} / action: True
[15:22:48 2024-02-23|INFO|main] 2-th step in 0-th episode / obs: (4, 8) / reward: 1.0 / done: False / info: {} / action: True
[15:22:48 2024-02-23|INFO|main] 3-th step in 0-th episode / obs: (8, 7) / reward: 4.0 / done: False / info: {} / action: True
[15:22:48 2024-02-23|INFO|main] 4-th step in 0-th episode / obs: (8, 7) / reward: 0.0 / done: False / info: {} / action: False
[15:22:48 2024-02-23|INFO|main] 5-th step in 0-th episode / obs: (8, 7) / reward: 8.0 / done: False / info: {} / action: True
[15:22:48 2024-02-23|INFO|main] 6-th step in 0-th episode / obs: (1, 6) / reward: 8.0 / done: False / info: {} / action: True
[15:22:48 2024-02-23|INFO|main] 7-th step in 0-th episode / obs: (8, 6) / reward: 0.0 / done: False / info: {} / action: False
[15:22:48 2024-02-23|INFO|main] 8-th step in 0-th episode / obs: (4, 5) / reward: 8.0 / done: False / info: {} / action: True
[15:22:48 2024-02-23|INFO|main] 9-th step in 0-th episode / obs: (0, 5) / reward: 0.0 / done: True / info: {} / action: False
"""
import logging
import numpy as np

from examples.access_control_q import *

# Setting logger
FORMAT = "[%(asctime)s|%(levelname)s|%(name)s] %(message)s"
DATE_FMT = "%H:%M:%S %Y-%m-%d"
log_level = logging.INFO
logging.basicConfig(format=FORMAT, datefmt=DATE_FMT, level=log_level)
logger = logging.getLogger('main')

# Environment configuration parameters.
NUM_STEPS = 10 #original is 100
NUM_SERVERS = 10
SERVER_FREE_PROB = 0.06
PRIORITIES = [1., 2., 4., 8.]

NUM_EPISODES = 1


def main():
    """Main routine of testing AccessControlQueue gym-type environment.
    """
    config = {'num_steps': NUM_STEPS,
              'num_servers': NUM_SERVERS,
              'server_free_probability': SERVER_FREE_PROB,
              'priorities': PRIORITIES}
    env = gym.make(id='AccessControlQueue-v0', config=config)
    for i in range(0, NUM_EPISODES):
        j = 0
        tmp = env.reset()
        log_reset(0, 0, tmp)
        while True:
            env.render()
            action = env.action_space.sample()  # Means random agent.
            obs, reward, done, info = env.step(action)
            log_step(i, j, obs, reward, done, info, action)
            j = j + 1
            if done:
                break
    env.close()


def log_step(episode: int, step: int, obs: np.ndarray, reward: float, done: bool, info: dict, action: int):
    """Utility function for printing logs.

    :param episode: Episode number.
    :param step: Time-step.
    :param obs: Observation of the current environment.
    :param reward: Reward from the current environment.
    :param done: Indicates whether the episode ends or not.
    :param info: Contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
    :param action: An action provided by the agent.
    """
    priority = obs[0].item()
    num_free_servers = obs[1].item()
    step_str = '{}-th step in {}-th episode / '.format(step, episode)
    obs_str = 'obs: {} / '.format((priority, num_free_servers))
    reward_str = 'reward: {} / '.format(reward)
    done_str = 'done: {} / '.format(done)
    info_str = 'info: {} / '.format(info)
    action_str = 'action: {}'.format(True if action else False)
    result_str = step_str + obs_str + reward_str + done_str + info_str + action_str
    logger.info(result_str)

def log_reset(episode: int, step: int, obs: np.ndarray):
    """Utility function for printing logs.

    :param episode: Episode number.
    :param step: Time-step.
    :param obs: Observation of the current environment.
    :param reward: Reward from the current environment.
    :param done: Indicates whether the episode ends or not.
    :param info: Contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
    :param action: An action provided by the agent.
    """
    capital = obs[0].item()
    step_str = '{}-th step in {}-th episode / '.format(step, episode)
    obs_str = 'obs: {} / '.format(capital)

    result_str = step_str + obs_str
    logger.info(result_str)

if __name__ == "__main__":
    main()
