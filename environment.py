"""
Module to represent game env, that is going to be used for RL
"""
import pickle
import numpy as np
from itertools import product
from typing import List, Tuple, Any
import random

WHITE_SQUARE, BLACK_SQUARE, BLACK_PIECE = 0, 1, 5
WHITE_PIECE, BLACK_KING, WHITE_KING = -5, 25, -25
ACTIVE_BLACK, ACTIVE_WHITE = 50, -50




def get_start_pieces_pos(start_row, end_row, size) -> List[Tuple[int, int]]:
    """
    Function to set up pieces on black squares
    at the very beginning of game. Tricky calculation allows to set up different
    boards, such as 6x6, 8x8, 10x10, in one function.

    Start and end rows (indexes) determine whether we are putting pieces from top
    or from bottom.

    """

    # As our board always has even side, size % 2 = 0.
    # For board of any size, first player has start_row = 0. So we'll start
    # setting the pieces from (1, 0). Unlike that, opposite side is always
    # different:
    # below you can see, that in loop we make some shifts
    # like "(index + 1) % 2". This version works fine for any board, where
    # start_row % 2 = 0 also. But for classic board 8x8 it does not.

    # Thus, we have to determine if size and start row have the
    # same parity, and use it as another shift variable:
    odd_shift = (size - start_row)

    result = []
    [
        [
            result.append(
                (row, column + (index + 1 - odd_shift) % 2)
            ) for column in range(0, size, 2)
        ] for index, row in enumerate(range(start_row, end_row))
    ]
    return result


def build_empty_board(size) -> np.array:
    """
    Create empty desc to start the game. We can get either numeric numpy
    array (for calculation, processing) or emoji array (for displaying it
    during the console game).
    """

    white_square = 0
    black_square = 1
    dtype = int

    board = np.full((size, size), fill_value=white_square, dtype=dtype)
    board[1::2, ::2], board[::2, 1::2] = black_square, black_square
    return board

def set_up_board(board, size) -> np.array:
    """
    Function to set up the empty board. Black squares (available to
    move to) will be represented as 1. First player piece """

    first_player_pieces = get_start_pieces_pos(start_row=0, end_row=int(-1 * size / 3 // 1 * -1), size=size)  # Round up
    # Note: 1 player occupies at least 1/3 of the board (2/6, 3/8, e.t.c)
    second_player_pieces = get_start_pieces_pos(start_row=int(2 * size / 3), end_row=size, size=size)

    for i in range(len(board)):
        for j in range(len(board[i])):
            if (i, j) in first_player_pieces:
                board[i][j] = 5
            elif (i, j) in second_player_pieces:
                board[i][j] = -5

    return board


def get_pieces_indexes(board, rank) -> list[tuple]:
    """
    Function to extract desired elements indexes. For example, indexes of
    first player pieces.
    :return: coordinates of pieces.
    """
    indexes: tuple[np.array] = np.where(board == rank)
    return list(zip(indexes[0], indexes[1]))


def get_moves(piece: tuple, step: int, backward=True) -> list[tuple]:
    """
    Function to get all possible ways to move the piece. After getting the
    result we are going the get REALLY POSSIBLE ways to move it.
    :param piece: our piece to move
    :param step: 1 square for simple move, 2 for takes over the opposite piece.
    For first player (human) step is positive, while for second it's negative
    :param backward: permission to move backwards (it's always true for Takes
    and false for moves if it's not King)

    """
    deltas = [step, -step]
    cart_prod = list(product(deltas, deltas))
    if not backward:
        cart_prod = cart_prod[:2]  # We take 2/4
    return [  # Our possible directions (at most four)
        (piece[0] + x, piece[1] + y)
        for x, y in cart_prod
    ]


def get_moves_list(
        board: np.array,
       piece: tuple[int, int],
       player_pointer = 1,
       is_king: bool =False
    ) -> List[tuple]:
    """
    Generator function to get all ways to capture. If any (for any piece), player
    if obliged to make it instead of simple move, so we check them all.
    """
    bound = board.shape[0]
    hypothetical_moves = get_moves(
        piece=piece,
        step=1 * player_pointer,
        backward=is_king
    )
    for move in hypothetical_moves:
        # We can't check all condition at once
        if (0 <= move[0] < bound) and (
                0 <= move[1] < bound):
            if board[move] == 1:
                yield move


def get_captures_list(board: np.array,
                      piece: tuple[int, int],
                      player_pointer = 1
    ) -> List[tuple]:
    """
    Generator function to get all ways to capture. If any (for any piece), player
    if obliged to make it instead of simple move, so we check them all.
    """
    # pieces: list = get_pieces_indexes(board=self.board, rank=5 * player_pointer)
    bound = board.shape[0]
    hypothetical_takes = get_moves(
        piece=piece,
        step=2 * player_pointer,
        backward=True
    )
    for take in hypothetical_takes:
        # We can't check all condition at once

        if (0 <= take[0] < bound) and (
                0 <= take[1] < bound):
            if board[take] == 1 and (
                board[
                    (
                        int((take[0] + piece[0]) / 2), int((take[1] + piece[1]) / 2)
                    )
                ] in (- 5 * player_pointer, - 25 * player_pointer)
            ):
                yield take


def choose_random_move(board, moves_list) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    Function to simulate move choosing by NN. Up for now it's just random
    choice
    """

    # place for future code...
    indexes: tuple[np.array] = np.where(board == board)
    # list(zip(indexes[0], indexes[1]))
    board = (lambda x: x)(board)  # Do something like we're really choosing carefully
    piece, move = random.choice(moves_list)
    return piece, move


def one_hot_encode_matrix(matrix: np.array) -> np.array:
    """
    Function to make one hot representation of matrix, by splitting it into several ones.

    """
    ones_matrix = np.zeros_like(matrix)
    ones_matrix[matrix == BLACK_SQUARE] = 1

    # create the second matrix
    first_player_matrix = np.zeros_like(matrix)
    first_player_matrix[matrix == BLACK_PIECE] = 1

    # As king piece is better that simple piece, why don't we just say it's
    # two times better?
    first_player_matrix[matrix == BLACK_KING] = 2

    # create the third matrix
    second_player_matrix = np.zeros_like(matrix)
    second_player_matrix[matrix == WHITE_PIECE] = 1
    second_player_matrix[matrix == WHITE_KING] = 2

    return np.stack((ones_matrix, first_player_matrix, second_player_matrix), axis=0)  # shape is 3xSIZExSIZE


def calculate_reward(game_state: np.array, next_game_state: np.array, side: int=1) -> int:
    """
    Function to return reward value
    :return: reward value
    """
    # to_do
    black_pieces_changes = next_game_state[1] - game_state[1]
    white_pieces_changes = next_game_state[2] - game_state[2]

    score_changes = (np.sum(black_pieces_changes),  np.sum(white_pieces_changes))[::side]  # Reverse
    # print(
    #     "SIDE:\n", side,
    #     "PLAYER REWARD: \n",
    #     score_changes[0] - score_changes[1], '\n',  # How much player earns AND How much opposite player lost
    #     (np.sum(black_pieces_changes), np.sum(white_pieces_changes)),
    #     "\n"
    # )
    return score_changes[0] - score_changes[1]  # How much player earns AND How much opposite player lost
    return (np.sum(black_pieces_changes),  np.sum(white_pieces_changes))[0]


def encode(matrix: list) -> int:
    # a, b, c, d = matrix.flatten()
    [a, b], [c, d] = matrix
    encoded_value = ((a * 8 + b) * 8 + c) * 8 + d
    return encoded_value


def decode(n) -> List[List[int]]:
    d = n % 8
    c = n // 8
    b = c // 8
    c = c % 8
    a = b // 8
    b = b % 8

    return (
        [[a, b], [c, d]]
    )


def get_all_moves_endoded(size: int = 8):
    """
    Function to collect ALL possible moves.
    I already know it's 172, but who's going to believe?
    """
    board = build_empty_board(size=size)
    all_moves = set()

    pieces: list = get_pieces_indexes(
        board=board,
        rank=1  # All black squares
    )
    for piece in pieces:

        hypothetical_moves = get_moves(
            piece=piece,
            step=1,
            backward=True
        )
        for move in hypothetical_moves:
            if (0 <= move[0] < size) and (
                    0 <= move[1] < size):
                all_moves.add(
                    encode([piece, move])
                )

        hypothetical_takes = get_moves(
            piece=piece,
            step=2,
            backward=True
        )
        for take in hypothetical_takes:
            if (0 <= take[0] < size) and (
                    0 <= take[1] < size):
                # if board[take] == 1:
                all_moves.add(
                    encode([piece, take])
                )

    return all_moves



def save_set(s: set) -> set:
    try:
        with open("set_of_moves.pickle", "rb") as file:
            old_set = pickle.load(file)
    except FileNotFoundError:
        old_set = set()
    s.update(old_set)
    with open("set_of_moves.pickle", "wb") as file:
        pickle.dump(s, file)

    print(
        "TOTAL NUMBER OF MOVES IS: ", len(s),
        # "\nMOVES ARE: \n", sorted(s)
          )
    return s




class CheckersEnvironment:
    def __init__(self, size=8):
        board = build_empty_board(size=size)
        self.board = set_up_board(board, size)
        self.size = size
        self.actions = set()

        # self.first_player = self.ComputerPlayer(game=self, side=1)
        # self.second_player = self.ComputerPlayer(game=self, side=-1)



    # class ComputerPlayer():
    #     def __init__(self, game, side=-1):
    #         """
    #         Initializing starts the game from scratch, defining the size and
    #         players pieces
    #          """
    #         self.game = game
    #         self.player_pointer = side
    #         self.picked = None


    def make_move(self, side: int = 1, desired_move: np.array or None = None) -> bool:
        """
        Function to move as a System player. Picks one piece and moves it.
        Firstly, we check all possible ways to capture the opponent. For
         all pieces. That's because of in case we can capture, we must do so.
         If no captures, we get all other moves, then select one.

        """

        board = np.copy(self.board)  # We need copy to compare the result

        pieces: list = get_pieces_indexes(
            board=board, rank=side * 5
        )
        kings: list = get_pieces_indexes(
            board=board, rank=side * 25
        )
        all_pieces = [*pieces, *kings]

        if not all_pieces:  # Empty
            # End of the game
            return False

        takes: list[tuple[tuple[int, int]]] = []  # Tuples of start and end position
        [
            [takes.append((piece, x)) for x in get_captures_list(
                board=board,
                piece=piece, player_pointer=side
            )] for piece in all_pieces
        ]

        if not takes:
            moves: list[tuple[tuple[int, int], tuple[int, int]]] = []  # Tuples of start and end position
            [
                [moves.append((piece, x)) for x in get_moves_list(
                    board=board,
                    piece=piece, player_pointer=side, is_king=(piece in kings))
                 ] for piece in all_pieces
            ]

        # Finally, our move must be one of these
        moves = takes or moves  # takes have priority
        k = False
        if moves:
            if desired_move:
                [a, b], [c, d] = desired_move
                if ((a,b), (c, d)) in moves:
                    # print("ХОД НЕ РАНДОМНЫЙ: ", desired_move)
                    piece, move = desired_move
                    k = True
            # print("Хотелось бы: ", desired_move)
            if not k:
                piece, move = choose_random_move(board=board, moves_list=moves)

            self.actions.add(encode([piece, move]))

        else:
            # End of the game
            return False

        self.picked = piece

        # Calvulate the index of bound. Here piece becomes the King
        bound =(self.size - 1) * (1 + side)  / 2
        board[move] = (25 * side if move[0] == bound else board[piece])
        board[piece] = 1

        if takes:  # If we captured some piece, we have to delete it
            opponent_piece = tuple(
                map(
                    lambda x: int(x / 2), [
                        (move[0] + piece[0]), (move[1] + piece[1])
                    ]
                )
            )
            board[opponent_piece] = 1



        self.board = board
        # self.game.update_board(new_board=board, player_pointer=self.player_pointer)


        return True



    def reset(self):
        self.__init__()
        return self.board

    def step(self, action: int, player: int=1) -> Tuple[np.array, int, bool]:
        # Применение действия к доске
        # Вернуть следующее состояние, вознаграждение, флаг окончания и дополнительную информацию

        # Here we transform move id to 2x2 matrix
        # TO-DO #
        board_before = self.board
        action = decode(action)
        # made_move = self.make_move(side=player, desired_move=np.array([[1,0], [2,1]]))
        made_move = self.make_move(side=player, desired_move=action)
        done = not made_move
        next_state = self.board

        reward = calculate_reward(
            game_state=one_hot_encode_matrix(board_before),
            next_game_state=one_hot_encode_matrix(next_state),
            side=player
        )

        return next_state, reward, done





env = CheckersEnvironment()
all_possible_moves = get_all_moves_endoded(size=8)

print(
    "LEN: ",
    len(all_possible_moves), "\n",
    # sorted(all_possible_moves)
)


for episode in range(1000):
        state = env.reset()
        # state = env.board
        done = False
        total_reward = 0
        player = 1
        while not done:
            # action = agent.act(state)
            next_state, reward, done = env.step(action=2723, player=player)
            # agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            player *= -1
        # agent.replay(batch_size)
        # print("TOTAL REWARD: ", total_reward)


setet = save_set(env.actions)
print("DIFFERENCE ARE: ", len(all_possible_moves - setet), len(setet - all_possible_moves))

# sets1 = save_set(set()) - sets
# sets1 = list(sets1)
# sets1.sort()
# print(sets1, "\n", len(sets1))
#
# sets =  sets - save_set(set())
# sets = list(sets)
# sets.sort()
# print(sets, "\n", len(sets))
# print(encode([[5,2], [4,3]]))
