from itertools import product
import numpy as np
import pandas as pd
from tabulate import tabulate
import random
from time import sleep

square_types = {
    0: '🟨',  # White square
    1: '⬛',  # Black square,
    5: '🟤',  # Black piece
    -5: '⚪',
    25: '🔴',  # Black King
    -25: '⭕',  # White King
    50: '🔵',  # Active Black
    -50: '🟢'  # Active White
}

def get_start_pieces_pos(start_row, end_row, size) -> list:
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



def build_empty_board(size, board_type='numeric') -> np.array:
    """
    Create empty desc to start the game. We can get either numeric numpy
    array (for calculation, processing) or emoji array (for displaying it
    during the console game).
    """
    if board_type == 'numeric':
        white_square = 0
        black_square = 1
        dtype = int
    elif board_type == 'emoji':
        white_square = square_types[0]
        black_square = square_types[1]
        dtype = str
    else:
        raise ValueError('Incorrect type of board')

    board = np.full((size, size), fill_value=white_square, dtype=dtype)
    board[1::2, ::2], board[::2, 1::2] = black_square, black_square

    return board


def set_up_board(board, size) -> np.array:
    """ Function to set up the empty board"""

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




def choose_best_move(board, moves_list) -> tuple[tuple[int, int], tuple[int, int]]:
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







class Game:
    def __init__(self, size=6, output='numeric'):
        """
        Initializing starts the game from scratch, defining the size and
        players pieces
         """
        self.size = size
        self.board_type = output  # Type of output


        board = build_empty_board(size=size, board_type='numeric')
        self.board = set_up_board(board, size)
        self.train_data = np.array(
            [
            [*self.board[np.nonzero(self.board)], *((0,) * 4)],
            ]
        )
        print(self.train_data)
        # self.picked_by_human = None
        # self.picked_by_pc = None

        # self.picked_by_first = None
        # self.picked_by_second = None

        # self.first_player = self.HumanPlayer(side=1, game=self)
        self.first_player = self.ComputerPlayer(game=self, side=1)
        self.second_player = self.ComputerPlayer(game=self, side=-1)

    def calculate_score(self, board_before: np.array, board_after: np.array, side: int = 1) ->tuple[int, int]:
        """
        Function to calculate player scores after his move.
        The better the move, the higher the score.

        :param board_before:
        :param board_after:
        :param side:
        :return:
        """
        # print(board_before - board_after)
        return (np.sum(board_after - board_before) * side, np.sum(board_after - board_before) * (-side))[::side]




    def put_to_dataset(self, move_result: np.array, player_pointer: int = 1) -> None:
        """
        Function to put the move information and result into the dataset.
        We add the current board status, the chosen move and result of
        this move (scores). We also add the pointer that shows which player
        made the move.
        """
        data = self.train_data
        board = self.board
        # print(board - move_result)
        scores = self.calculate_score(
            board_before = board,
            board_after = move_result,
            side=player_pointer,
        )
        print("SCORES: {}; {}".format(*scores))

        # We record, which player made the move
        whose_move = tuple(map(lambda x: max(x * player_pointer, 0), (1, -1)))

        # Finally, we add: 1) matrix description; 2) move priority; 3) scores
        one_round_data = (
            np.append(
                board[np.nonzero(board)],
                values = [*whose_move, *scores])
        )

        self.train_data = np.insert(
            data,
            obj=len(data),  # Put into the end of array
            values=one_round_data,
            axis=0
        )

        # self.train_data = np.insert(
        #     data,
        #     obj=len(data), # Put into the end of array
        #     values=[*board[np.nonzero(board)], score[0], score[1]],
        #     axis=0)


    def update_board(self, new_board: np.array, collect_data: bool = True, player_pointer: int = 1) -> None:
        """
        Function to update the game board (by replacing it with new one),
        and, if needed, to collect data for training
        """
        if collect_data:
            self.put_to_dataset(move_result=new_board, player_pointer=player_pointer)
        self.board = new_board


    class HumanPlayer():
        def __init__(self, game, side=1):
            """
            Initializing starts the game from scratch, defining the size and
            players pieces
             """
            self.game = game
            self.player_pointer = side
            self.picked = None

        def make_move(self) -> bool:
            """
            Function to make move as a human.
            Firstly, player input the coordinates of desired piece.
            By next step we got the square to move it on.
            When something's going wrong with the input, we call this function
            recursively to give to player another chance.
            """
            # print(self.game.get_current_board())
            pieces = get_pieces_indexes(board=self.game.board, rank=5)
            kings = get_pieces_indexes(board=self.game.board, rank=25)
            all_pieces = [*pieces, *kings]

            if not all_pieces:
                print("YOU LOST THE GAME")
                return False

            piece = input("Enter the coordinates of the piece you want to move (row, col): \n"
                          "Input 'END' to finish the game.\n")

            if piece.upper() == 'END':
                print("YOU LOST THE GAME")
                return False

            try:
                piece = tuple(map(int, piece.split(',')))
                # assert piece in get_pieces_indexes(board=self.board, rank=5), ValueError
                assert piece in all_pieces, ValueError
                # assert piece in self.human_pieces, ValueError
            except (ValueError, AssertionError):
                print('You did wrong input. Try again! ')
                # return self.game.make_human_move()
                return self.make_move()

            bound = self.game.size

            # Firstly, we check if player had to beat enemy piece somewhere.
            # In that case it's not allowed to make other moves
            takes: list[tuple[tuple[int, int]]] = []  # Tuples of start and end position
            [
                [takes.append(x) for x in self.game.get_captures_list(
                    piece, player_pointer=1
                )] for piece in all_pieces
            ]

            # If it's no way to take, just check other moves
            if not takes:
                # If piece is king, we can move it to both top and bottom
                # simple_moves = get_moves(piece=piece, step=1, backward=(piece in kings))

                moves: list[tuple[tuple[int, int], tuple[int, int]]] = []  # Tuples of start and end position
                moves = list(self.game.get_moves_list(
                    piece, player_pointer=1, is_king=(piece in kings))
                )


            # moves = takes or simple_moves  # Takes have a priority
            moves = takes or moves  # Takes have a priority
            if moves:

                # self.game.picked_by_human = piece
                self.picked = piece
                # print(self.game.get_current_board())
                # print(self.game.show_board())
                move = input("Enter the coordinates of the square you want to move to (row, col): ")
                try:
                    move = tuple(map(int, move.split(',')))

                    # Here we firstly will check for takes as it's actually obliged
                    assert move in moves, ValueError
                except (AssertionError, ValueError):
                    print('You did wrong input. Try again! ')
                    return self.make_move()
                    # return self.game.make_human_move()

                # If piece reaches the board, we make it a King.
                # Otherwise, no changes
                self.game.board[move] = (25 if move[0] + 1 == bound else self.game.board[piece])
                self.game.board[piece] = 1

                if move in takes:
                    enemy_piece = tuple(map(lambda x: int(x / 2), [(move[0] + piece[0]), (move[1] + piece[1])]))

                    self.game.board[enemy_piece] = 1
            else:
                print("NO WAY TO MOVE THIS PIECE! Try again! ")
                return self.make_move()
                # return self.game.make_human_move()
            self.game.train_data = np.append(
                self.game.train_data,
                [self.game.board[np.nonzero(self.game.board)]]
            )
            return True


    class ComputerPlayer():
        def __init__(self, game, side=-1):
            """
            Initializing starts the game from scratch, defining the size and
            players pieces
             """
            self.game = game
            self.player_pointer = side
            self.picked = None

        def make_move(self) -> bool:
            """
            Function to move as a System player. Picks one piece and moves it.
            Firstly, we check all possible ways to capture the opponent. For
             all pieces. That's because of in case we can capture, we must do so.
             If no captures, we get all other moves, then select one.

            """

            # print(self.game.get_current_board())
            # sleep(0.01)
            board = np.copy(self.game.board)  # We need copy to compare the result
            pieces: list = get_pieces_indexes(
                board=board, rank=self.player_pointer * 5
            )
            kings: list = get_pieces_indexes(
                board=board, rank=self.player_pointer * 25
            )
            all_pieces = [*pieces, *kings]

            if not all_pieces:  # Empty
                print("SKYNET LOST THE GAME. YOU'VE STOPPED THE MACHINE APOCALYPSE")
                return False

            takes: list[tuple[tuple[int, int]]] = []  # Tuples of start and end position
            [
                [takes.append((piece, x)) for x in self.game.get_captures_list(
                    piece, player_pointer=self.player_pointer
                )] for piece in all_pieces
            ]

            if not takes:
                moves: list[tuple[tuple[int, int], tuple[int, int]]] = []  # Tuples of start and end position
                [
                    [moves.append((piece, x)) for x in self.game.get_moves_list(
                        piece, player_pointer=self.player_pointer, is_king=(piece in kings))
                     ] for piece in all_pieces
                ]

            # Finally, our move must be one of these
            moves = takes or moves  # takes have priority
            if moves:
                piece, move = choose_best_move(board=board, moves_list=moves)
            else:
                print("SKYNET LOST THE GAME. YOU'VE STOPPED THE MACHINE APOCALYPSE")
                return False

            self.picked = piece

            print(self.game.get_current_board())
            sleep(0.01)

            # Calvulate the index of bound. Here piece becomes the King
            bound =(self.game.size - 1) * (1 + self.player_pointer)  / 2
            board[move] = (25 * self.player_pointer if move[0] == bound else board[piece])
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

            # self.game.train_data = np.append(
            #     self.game.train_data,
            #     [self.game.board[np.nonzero(self.game.board)]]
            # )
            # print(self.game.board[np.nonzero(self.game.board)])

            # self.game.train_data = np.insert(
            #     self.game.train_data,
            #     obj=len(self.game.train_data),
            #     values=[*self.game.board[np.nonzero(self.game.board)], 0],
            #     axis=0)

            # self.game.put_to_dataset()
            self.game.update_board(new_board=board, player_pointer=self.player_pointer)


            return True

    def get_current_board(self):
        board_type = self.board_type
        if board_type == 'numeric':
            board = self.board
        elif board_type == 'emoji':
            board = build_empty_board(self.size, board_type='emoji')


            for i in range(len(board)):
                for j in range(len(board[i])):
                    # Replace value by its picture representation
                    board[(i, j)] = square_types[self.board[(i, j)]]


            # if self.picked_by_human:
            #     board[self.picked_by_human[0]][self.picked_by_human[1]] = square_types[50]
            # elif self.picked_by_pc:
            #     board[self.picked_by_pc[0]][self.picked_by_pc[1]] = square_types[-50]



            # if self.picked_by_human:
            if self.second_player.picked:
                board[self.second_player.picked[0]][self.second_player.picked[1]] = square_types[50]
            elif self.first_player.picked:
                board[self.first_player.picked[0]][self.first_player.picked[1]] = square_types[-50]

            board = tabulate(board, showindex=True, headers=['row', *list([f'col {i} ' for i in range(self.size)])],
                             tablefmt="pretty")

        # self.picked_by_human = None
        # self.picked_by_pc = None
        self.first_player.picked = None
        self.second_player.picked = None

        return board


    def show_board(self):
        print(self.get_current_board())

    def get_moves_list(self, piece: tuple[int, int], player_pointer = 1, is_king: bool =False) -> list[tuple]:
        """
        Generator function to get all ways to capture. If any (for any piece), player
        if obliged to make it instead of simple move, so we check them all.
        """
        bound = self.size
        hypothetical_moves = get_moves(
            piece=piece,
            step=1 * player_pointer,
            backward=is_king
        )
        for move in hypothetical_moves:
            # We can't check all condition at once
            if (0 <= move[0] < bound) and (
                    0 <= move[1] < bound):
                if self.board[move] == 1:
                    yield move


    def get_captures_list(self, piece: tuple[int, int], player_pointer = 1
        ) -> list[tuple]:
        """
        Generator function to get all ways to capture. If any (for any piece), player
        if obliged to make it instead of simple move, so we check them all.
        """
        # pieces: list = get_pieces_indexes(board=self.board, rank=5 * player_pointer)
        bound = self.size
        hypothetical_takes = get_moves(
            piece=piece,
            step=2 * player_pointer,
            backward=True
        )
        for take in hypothetical_takes:
            # We can't check all condition at once

            if (0 <= take[0] < bound) and (
                    0 <= take[1] < bound):
                if self.board[take] == 1 and (
                    self.board[
                        (
                            int((take[0] + piece[0]) / 2), int((take[1] + piece[1]) / 2)
                        )
                    ] in (- 5 * player_pointer, - 25 * player_pointer)
                ):
                    yield take


    def play(self):
        # self.show_board()
        if self.first_player.make_move():
            # self.show_board()
            if self.second_player.make_move():
                return self.play()
        self.show_board()
        print(
            "RESULT: \n",
            # self.board,
            # self.board.ravel(),
            # self.board[np.nonzero(self.board)],
            pd.DataFrame(self.train_data), sep='\n\n')
        return print("THE END OF THE GAME")




if __name__ == "__main__":
    # game = Game(size=4, board_type='emoji')
    # print(game.get_current_board())
    game = Game(size=6, output='emoji')
    # play(game=game)
    # print( game.train_data)
    game.play()
    # print(game.get_current_board())
    # game = Game(size=8, board_type='emoji')
    # print(game.get_current_board())
    # game = Game(size=10, board_type='emoji')
    # print(game.get_current_board())
    # game = Game(size=6, board_type='numeric')



# Play the game
#     while True:
#         print(game.get_current_board())
#         # print(game.board)
#         game.make_human_move()
#         print('Computer is making a move...')
#         game.make_pc_move()
