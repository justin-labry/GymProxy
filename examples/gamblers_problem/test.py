# Author: Seungjae Shin <sjshin0505@{etri.re.kr, gmail.com}>

"""Testing script for GamblersProblem environment implemented based on the following reference:
R. S. Sutton and A. G. Barto, Reinforcement Learning - An Introduction, 2nd ed., 2018 (Example 4.3: Gambler's problem).
"""
"""
/home/labry/GymProxy/venv/bin/python /home/labry/GymProxy/examples/gamblers_problem/test.py 
[15:25:28 2024-02-23|INFO|main] 0-th step in 0-th episode / obs: 20 / reward: 0.0 / done: False / info: {'flip_result': 'head'} / action: 10
[15:25:28 2024-02-23|INFO|main] 1-th step in 0-th episode / obs: 40 / reward: 0.0 / done: False / info: {'flip_result': 'head'} / action: 20
[15:25:28 2024-02-23|INFO|main] 2-th step in 0-th episode / obs: 35 / reward: 0.0 / done: False / info: {'flip_result': 'tail'} / action: 5
[15:25:28 2024-02-23|INFO|main] 3-th step in 0-th episode / obs: 46 / reward: 0.0 / done: False / info: {'flip_result': 'head'} / action: 11
[15:25:28 2024-02-23|INFO|main] 4-th step in 0-th episode / obs: 0.0 / reward: 0.0 / done: True / info: {'flip_result': 'tail', 'msg': 'Loses the game due to out of money.'} / action: 46

/home/labry/GymProxy/venv/bin/python /home/labry/GymProxy/examples/gamblers_problem/test.py 
[15:26:51 2024-02-23|INFO|main] 0-th step in 0-th episode / obs: 20 / reward: 0.0 / done: False / info: {'flip_result': 'head'} / action: 10
[15:26:51 2024-02-23|INFO|main] 1-th step in 0-th episode / obs: 40 / reward: 0.0 / done: False / info: {'flip_result': 'head'} / action: 20
[15:26:51 2024-02-23|INFO|main] 2-th step in 0-th episode / obs: 80 / reward: 0.0 / done: False / info: {'flip_result': 'head'} / action: 40
[15:26:51 2024-02-23|INFO|main] 3-th step in 0-th episode / obs: 65 / reward: 0.0 / done: False / info: {'flip_result': 'tail'} / action: 15
[15:26:51 2024-02-23|INFO|main] 4-th step in 0-th episode / obs: 100.0 / reward: 1.0 / done: True / info: {'flip_result': 'head', 'msg': 'Wins the game because the capital becomes 100 dollars.'} / action: 65

"""
import logging
import numpy as np

from examples.gamblers_problem import *

# Setting logger
FORMAT = "[%(asctime)s|%(levelname)s|%(name)s] %(message)s"
DATE_FMT = "%H:%M:%S %Y-%m-%d"
log_level = logging.INFO
logging.basicConfig(format=FORMAT, datefmt=DATE_FMT, level=log_level)
logger = logging.getLogger('main')

# Environment configuration parameters.
NUM_STEPS = 100
PROB_HEAD = 0.5
INITIAL_CAPITAL = 10
WINNING_CAPITAL = 100

NUM_EPISODES = 1


def main():
    """Main routine of testing GamblersProblem gym-type environment.
    """
    config = {'num_steps': NUM_STEPS,
              'prob_head': PROB_HEAD,
              'initial_capital': INITIAL_CAPITAL,
              'winning_capital': WINNING_CAPITAL}

    metadata_ = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    env = gym.make(id='GamblersProblem-v0', config=config, metadata=metadata_)
    # obs = env.reset()
    # print(obs)
    for i in range(0, NUM_EPISODES):
        j = 0
        obs, info = env.reset(seed=123, options={})
        log_reset(0, 0, obs, info)
        while True:
            env.render()
            action = env.action_space.sample()  # Means random agent

            # Amount of betting should be less than current capital.
            action[0] = min(action[0].item(), obs[0].item())

            obs, reward, terminated, truncated, info = env.step(action)
            log_step(i, j, obs, reward, terminated, truncated, info, action)
            j = j + 1
            if terminated:
                break

    print("env.close()")
    env.close()


def log_step(episode: int, step: int, obs: np.ndarray, reward: float, terminated: bool, truncated: bool, info: dict, action: np.ndarray):
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
    bet = action[0].item()
    step_str = '{}-th step in {}-th episode / '.format(step, episode)
    obs_str = 'obs: {} / '.format(capital)
    reward_str = 'reward: {} / '.format(reward)
    terminated_str = 'terminated: {} / '.format(terminated)
    truncated_str = 'truncated: {} / '.format(truncated)
    info_str = 'info: {} / '.format(info)
    action_str = 'action: {}'.format(bet)
    result_str = step_str + obs_str + reward_str + terminated_str + truncated_str + info_str + action_str
    logger.info(result_str)

def log_reset(episode: int, step: int, obs: np.ndarray, info: dict):
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
    info_str = 'info: {} / '.format(info)
    step_str = '{}-th step in {}-th episode / '.format(step, episode)
    obs_str = 'obs: {} / '.format(capital)

    result_str = step_str + obs_str + info_str
    logger.info(result_str)


if __name__ == "__main__":
    main()
