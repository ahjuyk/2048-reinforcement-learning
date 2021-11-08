"""Gather training data from the game"""

# from __future__ import print_function

import argparse
import time
import random

import gym
import numpy as np
import pygame
import tensorflow as tf
from tensorflow.keras.models import load_model

import matplotlib
matplotlib.use("Agg")

import matplotlib.backends.backend_agg as agg
import matplotlib.pyplot as plt

import gym_2048
import training_data
import train_keras_model
from DQN_Model import DQN

grid_size = 70

class EndingEpisode(Exception):
    def __init__(self):
        super(EndingEpisode, self).__init__()

class Quitting(Exception):
    def __init__(self):
        super(Quitting, self).__init__()

def get_figure(width, height):
    dpi = 100.
    return plt.figure(figsize=[width / dpi, height / dpi], # Inches
                      dpi=100,        # 100 dots per inch, so the resulting buffer is 400x400 pixels
                     )

def get_bar_chart(fig, predictions):
    fig.clf()
    ax = fig.gca()
    ax.set_xlabel('Action')
    ax.set_ylabel('Probability')
    ax.set_ylim([0, 1])
    ax.bar(['Up', 'Right', 'Down', 'Left'], predictions)

    plt.tight_layout()
    canvas = agg.FigureCanvasAgg(fig)
    canvas.draw()
    renderer = canvas.get_renderer()
    raw_data = renderer.tostring_rgb()
    return raw_data

def get_line_plot(fig, results):
    fig.clf()
    ax = fig.gca()
    ax.set_xlabel('Episode')
    ax.set_ylabel('Score')
    ax.set_xlim([0, len(results)])
    ax.plot(range(len(results)), [r['Average score'] for r in results], label="Average score")
    ax.plot(range(len(results)), [r['Max score'] for r in results], label="Max score")
    ax.legend()

    plt.tight_layout()
    canvas = agg.FigureCanvasAgg(fig)
    canvas.draw()
    renderer = canvas.get_renderer()
    raw_data = renderer.tostring_rgb()
    return raw_data

def unstack(stacked, layers=16):
    """Convert a single 4, 4, 16 stacked board state into flat 4, 4 board."""
    representation = 2 ** (np.arange(layers, dtype=int) + 1)
    return np.sum(stacked * representation, axis=2)

def high_tile_in_corner(board):
    """Reports whether the a high tile >=64 is in the corner of (flat) board."""
    assert board.shape == (4, 4)
    highest_tile = np.amax(board)
    if highest_tile < 64:
        return False
    tiles_equal_to_highest = np.equal(board, np.full((4, 4), highest_tile))
    corners_equal_to_highest = tiles_equal_to_highest[[0, 0, -1, -1], [0, -1, 0, -1]]
    high_tile_in_corner = np.any(corners_equal_to_highest)
    #print(f"{board}, {highest_tile}, {tiles_equal_to_highest}, {corners_equal_to_highest}, {high_tile_in_corner}")
    return high_tile_in_corner

def training(agent, seed=None):
    env = gym.make('2048-v0')
    if seed:
        env.seed(seed)
    else:
        env.seed()
    episodes = 1000  # 训练1000次
    score_list = []  # 记录所有分数
    max_score = 0
    # print("observation:", observation)
    chart_height = 4 * grid_size
    chart_width = 4 * grid_size
    fig = get_figure(chart_width, chart_height)
    fig2 = get_figure(chart_width, chart_height)
    try:
        for i in range(episodes):
            # Loop around performing moves
            s = env.reset()
            env.render()
            while True:

                done = False
                score = 0

                board_array = env.render(mode='rgb_array')  # rgb_array
                board_surface = pygame.surfarray.make_surface(board_array)
                screen.blit(board_surface, (0, 0))

                predictions = agent.act(s)
                predicted_action = np.argmax(predictions)
                pygame.display.update()


                predicted_is_illegal = False
                env2 = gym.make('2048-v0')
                env2.reset()
                env2.set_board(unstack(s))
                (board2, _, _, info2) = env2.step(predicted_action)
                predicted_is_illegal = info2['illegal_move']
                if predicted_is_illegal:
                    print("***Predicted is illegal.***")



                high_in_corner_before = high_tile_in_corner(unstack(s))
                high_in_corner_after = high_tile_in_corner(unstack(board2))
                lost_high_corner = high_in_corner_before and not high_in_corner_after
                if lost_high_corner:
                    print("***Lost high corner tile.***")

                if predictions.shape != (4,):
                    predictions = [predictions, np.random.random()-1.0, np.random.random()-1.0, np.random.random()-1.0]
                action = predicted_action
                if predicted_is_illegal or lost_high_corner:
                    if predicted_is_illegal:
                        second_action = np.argsort(predictions)[2]
                        action = second_action
                        env2 = gym.make('2048-v0')
                        env2.reset()
                        env2.set_board(unstack(s))
                        (board2, _, _, info2) = env2.step(action)
                        predicted_is_illegal = info2['illegal_move']
                        if predicted_is_illegal:
                            second_action = np.argsort(predictions)[1]
                            action = second_action
                            env2 = gym.make('2048-v0')
                            env2.reset()
                            env2.set_board(unstack(s))
                            (board2, _, _, info2) = env2.step(action)
                            predicted_is_illegal = info2['illegal_move']
                            if predicted_is_illegal:
                                second_action = np.argsort(predictions)[0]
                                action = second_action
                                predicted_is_illegal = False
                print("Final Selected action {}".format(action))

                next_s, reward, done, info = env.step(action)

                agent.remember(s, action, next_s, reward, unstack(next_s))
                agent.train()
                ms = np.max(unstack(next_s))
                if np.max(unstack(next_s)) >= max_score:
                    max_score = np.max(unstack(next_s))
                    if np.max(unstack(next_s)) >= 512:
                        agent.save_model()
                    with open('./model/score.txt', 'w') as f:
                        f.write("max_score:" + str(max_score))
                    print(max_score)
                print("max score:", max_score, "!!!!!!!!!!!!!!!!!!!!")
                print("current score:", np.max(unstack(next_s)))
                print(unstack(next_s))

                score += reward
                s = next_s
                if done:
                    # Draw final board
                    env.render()
                    score_list.append(score)
                    print('episode:', i, 'score:', score, 'max:', max(score_list))
                    print("End of game")
                    break

        print("Ending episode...")
        agent.save_model()
    except EndingEpisode:
        print("Ending episode...")

def testing(agent, episodes=100, seed = None):
    env = gym.make('2048-v0')
    if seed:
        env.seed(seed)
    else:
        env.seed()

    score_count = {2:0,
                   4:0,
                   8:0,
                   16:0,
                   32:0,
                   64:0,
                   128:0,
                   256:0,
                   512:0,
                   1024:0,
                   2048:0,
                   4096:0,
                   8192:0
                   }
    score_list = []  # 记录所有分数
    max_score = 0
    # print("observation:", observation)
    chart_height = 4 * grid_size
    chart_width = 4 * grid_size
    fig = get_figure(chart_width, chart_height)
    fig2 = get_figure(chart_width, chart_height)
    try:
        for i in range(episodes):
            # Loop around performing moves
            s = env.reset()
            env.render()
            while True:

                done = False
                score = 0

                board_array = env.render(mode='rgb_array')  # rgb_array
                board_surface = pygame.surfarray.make_surface(board_array)
                screen.blit(board_surface, (0, 0))

                predictions = agent.act(s)
                predicted_action = np.argmax(predictions)
                pygame.display.update()

                predicted_is_illegal = False
                env2 = gym.make('2048-v0')
                env2.reset()
                env2.set_board(unstack(s))
                (board2, _, _, info2) = env2.step(predicted_action)
                predicted_is_illegal = info2['illegal_move']
                if predicted_is_illegal:
                    print("***Predicted is illegal.***")

                high_in_corner_before = high_tile_in_corner(unstack(s))
                high_in_corner_after = high_tile_in_corner(unstack(board2))
                lost_high_corner = high_in_corner_before and not high_in_corner_after
                if lost_high_corner:
                    print("***Lost high corner tile.***")

                if predictions.shape != (4,):
                    predictions = [predictions, np.random.random() - 1.0, np.random.random() - 1.0,
                                   np.random.random() - 1.0]
                action = predicted_action
                if predicted_is_illegal or lost_high_corner:
                    if predicted_is_illegal:
                        second_action = np.argsort(predictions)[2]
                        action = second_action
                        env2 = gym.make('2048-v0')
                        env2.reset()
                        env2.set_board(unstack(s))
                        (board2, _, _, info2) = env2.step(action)
                        predicted_is_illegal = info2['illegal_move']
                        if predicted_is_illegal:
                            second_action = np.argsort(predictions)[1]
                            action = second_action
                            env2 = gym.make('2048-v0')
                            env2.reset()
                            env2.set_board(unstack(s))
                            (board2, _, _, info2) = env2.step(action)
                            predicted_is_illegal = info2['illegal_move']
                            if predicted_is_illegal:
                                second_action = np.argsort(predictions)[0]
                                action = second_action
                                predicted_is_illegal = False
                print("Final Selected action {}".format(action))

                next_s, reward, done, info = env.step(action)

                if np.max(unstack(next_s)) >= max_score:
                    max_score = np.max(unstack(next_s))
                    print(max_score)

                print("max score:", max_score, "!!!!!!!!!!!!!!!!!!!!")
                print("current score:", np.max(unstack(next_s)))
                print(unstack(next_s))


                score += reward
                s = next_s
                if done:
                    # Draw final board
                    ms = np.max(unstack(next_s))
                    score_count[ms] += 1
                    env.render()
                    score_list.append(score)
                    print('episode:', i, 'score:', score, 'max:', ms)
                    print("End of game")
                    break

        print("Ending episode...")
        agent.save_model()
        for i,k in enumerate(list(score_count.keys())):
            print(str(k),":",score_count[k],"次")
    except EndingEpisode:
        print("Ending episode...")

def get_reward_for_state_action(env, state, action):
    env.reset()
    env.set_board(state)
    new_observation, reward, done, info = env.step(action)
    return reward
def add_rewards_to_training_data(env, input_training_data):
    new_training_data = training_data.training_data()
    for n in range(input_training_data.size()):
        (state, action) = input_training_data.get_n(n)
        reward = get_reward_for_state_action(env, state, action)
        new_training_data.add(state, action, reward)
    return new_training_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default=None, help="Pre-trained model to start from (optional)")
    parser.add_argument('--seed', type=int, default=None, help="Set the seed for the game")
    args = parser.parse_args()

    if args.model:
        print("load model...")
        print(args.model)
        agent = DQN()
        model = load_model(str(args.model))
        agent.set_model(model)
    else:
        agent = DQN()

    # Initialise pygame for detecting keypresses
    pygame.init()
    height = 4 * grid_size
    width = 12 * grid_size
    screen = pygame.display.set_mode((width, height), 0, 32)
    pygame.font.init()
    # Initialise environment

    try:
        #training(agent)
        testing(agent, episodes=100)
    except Quitting:
        print("Quitting...")

    # Close the environment


