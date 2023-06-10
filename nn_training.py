import numpy as np
import tensorflow as tf
from game import Game
from copy import copy
import os

def load_weights(policy_model, value_model, load_path):
    policy_model.load_weights(load_path + '/policy_weights')
    value_model.load_weights(load_path + '/value_weights')


# Define the Policy Network
def policy_network(input_shape, output_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=input_shape),
        tf.keras.layers.Dense(64, activation='relu'),
        # tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(output_shape, activation='softmax')
    ])
    return model

# Define the Value Network
def value_network(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='tanh')
    ])
    return model

# Define a function to predict the best move
def predict_move(policy_model, game_state):
    # Reshape the game state to match the input shape of the policy network
    input_state = np.reshape(game_state, (1,) + game_state.shape)
    # Predict the probabilities of each possible move
    move_probabilities = policy_model.predict(input_state)[0]
    # Choose the move with the highest probability
    move_index = np.argmax(move_probabilities)
    # Convert the move index to a tuple of coordinates
    from_coord = (move_index // 64, (move_index % 64) // 8)
    to_coord = ((move_index % 64) % 8, move_index % 8)
    return (from_coord, to_coord)

# Define a function to calculate the reward
def calculate_reward(old_game_state, new_game_state):
    # Calculate the difference between the number of pieces in the old and new game states
    piece_difference = np.sum(old_game_state[:,:,1]) - np.sum(new_game_state[:,:,1])
    # If the piece difference is positive, the model captured a piece and should receive a positive reward
    if piece_difference > 0:
        reward = 1
    else:
        reward = 0
    return reward

# Define a function for reinforcement training
def train_model(policy_model, value_model, game_states, rewards, learning_rate):
    # Reshape the game states to match the input shape of the policy and value networks
    # game_states = np.array(game_states)
    # input_states = np.reshape(game_states, (game_states.shape[0],) + game_states.shape)
    print(game_states[0], type(game_states[0]))
    # return
    # Calculate the predicted values and gradients of the value network
    with tf.GradientTape() as tape:
        predicted_values = value_model(input_states)
        value_loss = tf.keras.losses.mean_squared_error(rewards, predicted_values)
        value_gradients = tape.gradient(value_loss, value_model.trainable_variables)
    # Update the weights of the value network
    value_optimizer =  tf.keras.optimizers.Adam(learning_rate=learning_rate)
    value_optimizer.apply_gradients(zip(value_gradients, value_model.trainable_variables))
    # Calculate the predicted probabilities and gradients of the policy network
    with tf.GradientTape() as tape:
        predicted_probabilities = policy_model(input_states)
        policy_loss = tf.keras.losses.sparse_categorical_crossentropy(np.argmax(predicted_probabilities, axis=1),
                                                                      rewards)
        policy_gradients = tape.gradient(policy_loss, policy_model.trainable_variables)
    # Update the weights of the policy network
    policy_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    policy_optimizer.apply_gradients(zip(policy_gradients, policy_model.trainable_variables))



def calculate_score(board_before: np.array, board_after: np.array, side: int = 1) ->tuple[int, int]:
    """
    Function to calculate player scores after his move.
    The better the move, the higher the score.
    We actually have to work only with scores player gets after HIS move,
    so for game of X rounds we will have X train records, not 2X.

    :param board_before:
    :param board_after:
    :param side:
    :return:
    """
    # print(board_before - board_after)
    # print(one_hot_encode_matrix(board_before))
    piece = 5 * side
    king = 25 * side
    reward = np.sum(board_after == piece) - np.sum(board_before == piece)
    reward += np.sum(board_after == king) - np.sum(board_before == king) * 2  # Kings are more important
    return reward


def train_on_one_game(board_history: list[np.array], learning_rate, save_path='', winner=1):
    """
    Function to train model on one game. For each game, we have
    some board states and exactly one winner (or draw). So, one game is one butch
    """
    # Define the input and output shapes of the networks
    # input_shape = (8, 8, 3)
    input_shape = (8, 8, 1)
    output_shape = 64 * 64
    # Initialize the policy and value networks
    policy_model = policy_network(input_shape, output_shape)
    value_model = value_network(input_shape)
    try:
        load_weights(policy_model, value_model, save_path)
    except:
        pass

    player = 1  # Player pointer (first or second)
    game_states = copy(board_history)
    rewards = []
    # Loop over the game's steps
    while board_history:
        # Move from end to begining
        board_after = board_history.pop()
        try:
            board_before = board_history[-1]
        except IndexError:
            # Only one board (first) left. Finish the training
            break

        reward = calculate_score(
            board_before=board_before,
            board_after=board_after,
            side=player
        )
        rewards.append(reward)
        player *= -1  # Reverse

    # Train the policy and value networks using the game states and rewards
    train_model(policy_model, value_model, game_states, rewards, learning_rate)
    # Save the weights of the trained networks
    policy_model.save_weights(save_path + '/policy_weights')
    value_model.save_weights(save_path + '/value_weights')


if __name__ == "__main__":
    for _ in range(3):
        game = Game(size=8, output='emoji')
        train_batch = game.play()
        #
        # train_process(train_batch, 0.01, batch_size=3, num_episodes=10, save_path='')
        train_on_one_game(board_history=train_batch, learning_rate=0.001, save_path=os.getcwd() + "/weights/")
        print("Weights are updated.")