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

def gather_training_data(env, model, data, results, max_score, seed=None):
    """Gather training data from letting the user play the game"""
    # Initialise seed for environment
    if seed:
        env.seed(seed)
    else:
        env.seed()
    observation = env.reset()
    # print("observation:", observation)
    chart_height = 4 * grid_size
    chart_width = 4 * grid_size
    fig = get_figure(chart_width, chart_height)
    fig2 = get_figure(chart_width, chart_height)
    print("User cursor keys to play, q to quit")
    try:
        while True:
            # Loop around performing moves
            action = None
            env.render()
            done = False

            board_array = env.render(mode='rgb_array')  # rgb_array
            board_surface = pygame.surfarray.make_surface(board_array)
            screen.blit(board_surface, (0, 0))

            # Get predictions from model
            predictions = model.predict(np.reshape(observation.astype('float32'), (-1, 256))).reshape((4))
            predicted_action = np.argmax(predictions)
            #print(predictions)

            # Report predicted rewards for actions
            dir_dict = {0: 'up', 1: 'right', 2: 'down', 3: 'left'}
            dir_reward = [(dir_dict[i], p) for i, p in enumerate(list(predictions))]
            dir_reward.sort(key=lambda x: x[1], reverse=True)
            for direction, reward in dir_reward:
                print('{}: {:.3f}'.format(direction, reward))

            # Create graph of predictions
            raw_data = get_bar_chart(fig, predictions)
            surf = pygame.image.fromstring(raw_data, (chart_height, chart_width), "RGB")
            screen.blit(surf, (4 * grid_size, 0))

            # Create graph of results
            raw_data2 = get_line_plot(fig2, results)
            surf2 = pygame.image.fromstring(raw_data2, (chart_height, chart_width), "RGB")
            screen.blit(surf2, (8 * grid_size, 0))

            pygame.display.update()

            # Ask user for action
            record_action = True
            # Auto-select best action according to model
            # Require at least 50% confidence
            # Naive view of confidence, not counting symmetrical boards
            confidence = np.max(predictions)
            if confidence < 0.5:
                print("***Confidence < 50%: {}***".format(confidence))

            predicted_is_illegal = False
            env2 = gym.make('2048-v0')
            env2.reset()
            env2.set_board(unstack(observation))
            (board2, _, _, info2) = env2.step(predicted_action)
            predicted_is_illegal = info2['illegal_move']
            if predicted_is_illegal:
                print("***Predicted is illegal.***")

            high_in_corner_before = high_tile_in_corner(unstack(observation))
            high_in_corner_after = high_tile_in_corner(unstack(board2))
            lost_high_corner = high_in_corner_before and not high_in_corner_after
            if lost_high_corner:
                print("***Lost high corner tile.***")
            action = predicted_action

            ''' 從手動選擇變成model挑選違規以外的動作 '''
            if confidence < 0.5 or predicted_is_illegal or lost_high_corner:
                if confidence < 0.5:
                    action = predicted_action
                if predicted_is_illegal:
                    second_action = np.argsort(predictions)[-2]
                    action = second_action
                    print("select act2", action)
                    env2 = gym.make('2048-v0')
                    env2.reset()
                    env2.set_board(unstack(observation))
                    (board2, _, _, info2) = env2.step(action)
                    predicted_is_illegal = info2['illegal_move']
                    if predicted_is_illegal:
                        second_action = np.argsort(predictions)[-3]
                        action = second_action
                        print("select act3", action)
                        env2 = gym.make('2048-v0')
                        env2.reset()
                        env2.set_board(unstack(observation))
                        (board2, _, _, info2) = env2.step(action)
                        predicted_is_illegal = info2['illegal_move']
                        if predicted_is_illegal:
                            second_action = np.argsort(predictions)[0]
                            action = second_action
                            print("select act4", action)
                            predicted_is_illegal = False
            print("Final Selected action {}".format(action))

            # Add this data to the data collection if manually entered and not illegal
            new_observation, reward, done, info = env.step(action)
            illegal_move = info['illegal_move']
            if record_action and not illegal_move:

                # Unstack the stacked state
                data.add(unstack(observation), action, reward, unstack(new_observation), done)
                print("max!!!!!!!!!!!!!!!!!!!!!:")
                print(np.max(unstack(new_observation)))

                # Train model using new data
                train_from_me = data.copy()
                train_from_me.augment()
                minibatch_size = min(32, train_from_me.size())
                sample_indexes = random.sample(range(train_from_me.size()), minibatch_size)
                sample_data = train_from_me.sample(sample_indexes)

                train_data = np.reshape(sample_data.get_x_stacked().astype('float'), (-1, board_size * board_size * board_layers))
                train_labels = sample_data.get_y_digit()

                history = model.fit(train_data,
                          train_labels,
                          epochs=1,
                          batch_size=32)
                '''儲存max score的model'''
                if np.max(unstack(new_observation)) >= max_score:
                    max_score = np.max(unstack(new_observation))
                    model.save('./model/2048model.hdf5')
                    print(history.history['loss'])
                    print(history.history['accuracy'])
                    with open('./model/score.txt', 'w') as f:
                        f.write("max_score:"+str(max_score)+"\n"+ "loss:"+str(history.history['loss'][0])+"\n"+"acc:"+str(history.history['accuracy'][0]))
            else:
                print("Not recording move")

            observation = new_observation
            print()

            if done:
                # Draw final board
                env.render()
                print("End of game")
                break
    except EndingEpisode:
        print("Ending episode...")

    return data, max_score


board_size = 4
board_layers = 16 # Layers of game board to represent different numbers
outputs = 4

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', default=None, help="Input existing training data to start from (optional)")
    parser.add_argument('--model', '-m', default=None, help="Pre-trained model to start from (optional)")
    parser.add_argument('--output', '-o', default='data_{}.csv'.format(int(time.time())), help="Set the output file name")
    parser.add_argument('--seed', type=int, default=None, help="Set the seed for the game")
    args = parser.parse_args()
    # Initialise environment
    env = gym.make('2048-v0')

    if args.model:
        print("load model...")
        model = load_model(args.model)
    else:
        filters = 64
        residual_blocks = 8
        model = train_keras_model.build_model(board_size, board_layers, outputs, filters, residual_blocks)
        model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

    # Initialise pygame for detecting keypresses
    pygame.init()
    height = 4 * grid_size
    width = 12 * grid_size
    screen = pygame.display.set_mode((width, height), 0, 32)
    pygame.font.init()
    alldata = training_data.training_data()
    if args.input:
        alldata.import_csv(args.input)


    results = [train_keras_model.evaluate_model(model, 10, 0.)]
    try:
        score = 0
        while True:
            _,score = gather_training_data(env, model, alldata, results, score, seed=args.seed,)
            results.append(train_keras_model.evaluate_model(model, 10, 0.))

            print("Got {} data values".format(alldata.size()))

    except Quitting:
        print("Quitting...")

    # Close the environment
    env.close()

    if alldata.size():
        alldata.export_csv(args.output)
