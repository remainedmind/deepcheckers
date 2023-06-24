import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F


import numpy as np
# import keras
# from keras.layers import Dense, Input, Flatten, Dropout
# from keras.models import Model
from game import Game


class CheckersModel(nn.Module):
    def __init__(self):
        super(CheckersModel, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(3, 128)  # Additional layer for features
        self.fc3 = nn.Linear(256, 4)
        self.rewards = np.array([])

    def forward(self, x, features):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        f = self.fc2(features)
        x = torch.cat((x, f), dim=1)  # Concatenate the two feature vectors
        x = torch.relu(x)
        x = self.fc3(x)
        return x


def get_move(game_state: np.array, next_game_state: np.array) -> np.array:
    """
    Function to compare to states and extract the made move
    (by considering the difference)
    :return: 2x2 numpy array
    """
    # print(next_game_state, game_state)
    diff = next_game_state - game_state
    indices = np.where(diff != 0)
    move = np.array([[indices[0][0], indices[1][0]], [indices[0][1], indices[1][1]]])
    return move


def calculate_reward(game_state: np.array, next_game_state: np.array, side: int=1) -> int:
    """
    Function to return reward value
    :return: reward value
    """
    # to_do
    black_pieces_changes = next_game_state[1] - game_state[1]
    white_pieces_changes = next_game_state[2] - game_state[2]
    print(
        "MOVE RESULT:\n",
        # (black_pieces_changes, white_pieces_changes)[min(side, 0)],
        black_pieces_changes,
        '\n-----------------------\n',
        white_pieces_changes,
        '\n\n'
    )
    score_changes = np.sum(black_pieces_changes),  np.sum(white_pieces_changes)
    print(
        "SIDE:\n", side,
        "PLAYER REWARD: \n",
        - score_changes[max(side, 0)], '\n',  # How much opposite player lost
        score_changes[min(side, 0)],  # How much player earns
          "\n")
    return np.sum(black_pieces_changes),  np.sum(white_pieces_changes)
    self.reward.append([])

    # print(
    #     (np.sum(black_pieces_changes), np.sum(white_pieces_changes)*(-side))[::side]
    # )
    # print(game_state[np.where(next_game_state[1:] - game_state[1:]) != 0])
    # print(next_game_state[1:] - game_state[1:])
    # print(diff.shape, diff)
    # for
    return 1
    pass


def collect_second_layer(game_states: list[np.array], winner: int) -> np.array:
    """
    Function to iterate over game states and collect all additional features.
    We assume that first player always moves first, so, we can just use pointer
    :return: array for second layer of model
    """
    player_pointer = 1
    num_states = len(game_states)
    second_layer_data = np.zeros((num_states, 4), dtype=np.int32)

    for i, state in enumerate(game_states[:-1]):
        first_rewards, second_rewards = calculate_reward(game_state=state, next_game_state=game_states[i+1], side=player_pointer)
        player_pointer *= -1 # Change to second player
        second_layer_data[i] = [player_pointer, winner, first_rewards, second_rewards]
    return second_layer_data



def collect_target_variables(game_states: np.array) -> np.array:
    """
    Function to iterate over game states and collect all additional features.
    We assume that first player always moves first, so, we can just use pointer
    :param game_states: array with shape (N, 3, 8, 8)
    :return: array for second layer of model
    """

    num_states = len(game_states)
    target_data = np.zeros((num_states - 1, 2, 2), dtype=np.int32)


    for i, state in enumerate(game_states[:-1]):
        move: np.array = get_move(game_state=state, next_game_state=game_states[i+1])

        # YOUR CODE TO ADD the move to variable vector (np.array).

        target_data[i] = move

    return target_data




if __name__ == "__main__":
    game = Game(size=8, output='emoji')
    NUMBER_OF_GAMES = 2
    batch_size = 10  # Set a fixed batch size
    num_epochs = 10  # Adjust as needed

    model = CheckersModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for _ in range(NUMBER_OF_GAMES):
        game = Game(size=8, output='emoji')
        train_batch: list[np.array]
        winner: int
        train_batch, winner = game.play()

        # train_batch = train_batch[:10]


        train_batch: np.array = np.stack(train_batch, axis=0)
        # print(train_batch.shape)
        second_layer: np.array = collect_second_layer(game_states=train_batch, winner=winner)

        # print(train_batch[:, 2, :, :])
        # continue

        # for state, record in zip(train_batch, second_layer):
        #     print("STATE: \n", state, "\nDATA: \n", record, "\n\n")


        # target_data = collect_target_variables(game_states=train_batch)
        # print(target_data)

        break


        # Convert the data to PyTorch tensors
        # game_states_tensor = torch.from_numpy(train_batch).float()
        # second_layer_tensor = torch.from_numpy(second_layer).float()
        # target_tensor = torch.from_numpy(target_data).long()

        # Convert the data to PyTorch tensors
        game_states_tensor = torch.from_numpy(train_batch[:-1]).float()
        game_states_tensor = game_states_tensor.permute(0, 3, 1, 2)  # Rearrange dimensions
        second_layer_tensor = torch.from_numpy(second_layer[1:]).float()
        target_tensor = torch.from_numpy(target_data).long()

        # Combine game states and second layer tensors into a dataset
        dataset = data.TensorDataset(game_states_tensor, second_layer_tensor, target_tensor)

        # Determine the number of batches and trim the dataset to match
        num_batches = len(dataset) // batch_size
        dataset = dataset[:num_batches * batch_size]

        # Create a data loader with the specified batch size
        dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)



        for epoch in range(num_epochs):
            for batch_states, batch_second_layer, batch_targets in dataloader:
                # Forward pass
                outputs = model(batch_states, batch_second_layer)
                loss = criterion(outputs, batch_targets.view(-1))

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


