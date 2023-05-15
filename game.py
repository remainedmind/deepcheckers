from itertools import product
import numpy as np
from tabulate import tabulate
import random
from time import sleep

square_types = {
    0: '🟨',  # White square
    1: '⬛',  # Black square,
    5: '🟤',
    -5: '⚪',
    25: '🔴',  # Black King
    -25: '🟠',
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
    """ Create empty desc to start the game. """
    if board_type == 'numeric':
        white_square = 0
        black_square = 1
        dtype = int
        # black_piece = 5
        # white_piece = -5
    elif board_type == 'emoji':
        white_square = square_types[0]
        black_square = square_types[1]
        dtype = str
        # black_piece = square_types[5]
        # white_piece = square_types[-5]
    else:
        raise ValueError('Incorrect type of board')

    board = np.full((size, size), fill_value=white_square, dtype=dtype)
    board[1::2, ::2], board[::2, 1::2] = black_square, black_square

    return board


def set_up_board(board, size) -> np.array:
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


# def make_move()




def choose_best_move(board, moves_list) -> tuple[tuple[int, int], tuple[int, int]]:
    """
    Function to simulate move choosing by NN. Up for now it's just random
    choice
    """

    # place for future code...
    board = (lambda x: x)(board)  # Do something like we're really choosing carefully
    piece, move = random.choice(moves_list)
    return piece, move





class Game:
    def __init__(self, size=6, board_type='emoji'):
        """
        Initializing starts the game from scratch, defining the size and
        players pieces
         """
        self.size = size
        self.board_type = board_type

        board = build_empty_board(size=size, board_type='numeric')
        self.board = set_up_board(board, size)
        # Set up the pieces for the start:
        self.human_pieces = get_start_pieces_pos(start_row=0, end_row=int(-1 * size / 3 // 1 * -1), size=size)
        # Note: 1 player occupies 1/3 of the board.
        self.computer_pieces = get_start_pieces_pos(start_row=int(2 * size / 3), end_row=size, size=size)
        self.picked_by_human = None
        self.picked_by_pc = None


    def pick_piece(self, player: str, piece: tuple[int, int]):
        """
        Actually only human needs picking, not a computer.
        However, to make game more handy, we will show, which piece if
        picked by PC
        """
        if player == 'human':
            if piece in self.human_pieces:
                self.picked_by_human = piece
                return
        elif player == 'pc':
            if piece in self.computer_pieces:
                self.picked_by_pc = piece
                return
        raise ValueError('Wrong piece attributes or wrong player type! ')


    def get_current_board(self):
        board_type = self.board_type
        if board_type == 'numeric':
            board = self.board
            for i in range(len(board)):
                for j in range(len(board[i])):
                    if (i, j) in self.human_pieces:
                        board[i][j] = 5
                    elif (i, j) in self.computer_pieces:
                        board[i][j] = -5

            if self.picked_by_human:
                board[self.picked_by_human[0]][self.picked_by_human[1]] = 50
            elif self.picked_by_pc:
                board[self.picked_by_pc[0]][self.picked_by_pc[1]] = -50


        elif board_type == 'emoji':
            board = build_empty_board(self.size, board_type='emoji')

            for i in range(len(board)):
                for j in range(len(board[i])):
                    # Replace value by its picture representation
                    board[(i, j)] = square_types[self.board[(i, j)]]

            if self.picked_by_human:
                board[self.picked_by_human[0]][self.picked_by_human[1]] = square_types[50]
            elif self.picked_by_pc:
                board[self.picked_by_pc[0]][self.picked_by_pc[1]] = square_types[-50]

            board = tabulate(board, showindex=True, headers=['row', *list([f'col {i} ' for i in range(self.size)])],
                             tablefmt="pretty")

        self.picked_by_human = None
        self.picked_by_pc = None
        return board


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


    # def get_captures_list(self, pieces: list[tuple], player_pointer = 1) -> list[tuple]:
    #     """
    #     Function to get all ways to capture. If any (for any piece), player
    #     if obliged to make it instead of simple move, so we check them all.
    #     """
    #     # pieces: list = get_pieces_indexes(board=self.board, rank=5 * player_pointer)
    #     bound = self.size
    #     # def check_capture(take, piece)
    #     for piece in pieces:
    #         takes = get_moves(piece=piece, step=2 * player_pointer)
    #         for take in takes:
    #             # We can't check all condition at once
    #             if (0 <= take[0] < bound) and (
    #                     0 <= take[1] < bound):
    #                 # print(len(self.size))
    #                 if self.board[take] == 1 and (
    #                     self.board[
    #                         (
    #                             int((take[0] + piece[0]) / 2), int((take[1] + piece[1]) / 2)
    #                         )
    #                     ] in (- 5 * player_pointer, - 25 * player_pointer)
    #                 ):
    #                     yield take


    def get_captures_list(self, piece: tuple[int, int], player_pointer = 1) -> list[tuple]:
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
                # print(len(self.size))
                if self.board[take] == 1 and (
                    self.board[
                        (
                            int((take[0] + piece[0]) / 2), int((take[1] + piece[1]) / 2)
                        )
                    ] in (- 5 * player_pointer, - 25 * player_pointer)
                ):
                    yield take


    # def get_move(self, player_pointer = 1) -> list[tuple]:
    #     """ Function to make move as a one of players. Pick one piece and move it.
    #     :argument attempts -  It's a marker of that PC is losing game. Read below
    #     :argument player - 1 or -1 (first player or second one)
    #
    #     """
    #     pieces: list = get_pieces_indexes(board=self.board, rank= 5 * player_pointer)
    #     if pieces:  # Not empty
    #         piece = random.choice(pieces)
    #     else:
    #         return []
    #
    #     bound = self.size
    #
    #     # While capturing the enemy's piece, we can move to both top and bottom
    #     takes = get_moves(piece=piece, step=-2)
    #
    #     takes = [
    #         take for take in takes if (  # We can't check all condition in one "all"
    #             0 <= take[0] < bound
    #             ) and (0 <= take[1] < bound
    #             ) and all(
    #             (
    #                 self.board[take] == 1,
    #                 self.board[
    #                     (
    #                         int((take[0] + piece[0]) / 2), int((take[1] + piece[1]) / 2)
    #                     )
    #                 ] in (5 * player_pointer, 25 * player_pointer))
    #         )
    #     ]
    #
    #     # If it's no way to take, just check other moves
    #     if not takes:
    #         hypothetical_moves = get_moves(piece=piece, step=-1, backward=False)
    #
    #         real_moves = [
    #             move for move in hypothetical_moves if 0 <= move[0] < bound and 0 <= move[1] < bound and self.board[move] == 1
    #         ]
    #
    #     moves = takes or real_moves
    #     return moves





    def make_pc_move(self, attempts=1) -> None:
        """
        Function to move as a System player. Picks one piece and moves it.
        Firstly, we check all possible ways to capture the opponent. For
         all pieces. That's because of in case we can capture, we must do so.
         If no captures, we get all other moves, then select one.
        :argument attempts -  It's a marker of that PC is losing game. Read below

        """


        pieces: list = get_pieces_indexes(board=self.board, rank= 5 * -1)

        if not pieces:  # Empty
            return False

        takes: list[tuple[tuple[int, int]]] = []  # Tuples of start and end position
        [
            [takes.append((piece, x)) for x in self.get_captures_list(
            piece, player_pointer=-1
        )] for piece in pieces
        ]

        if not takes:
            moves: list[tuple[tuple[int, int], tuple[int, int]]] = []  # Tuples of start and end position
            [
                [moves.append((piece, x)) for x in self.get_moves_list(
                    piece, player_pointer=-1, is_king=False)
                ] for piece in pieces
            ]

        # Finally, our move must be one of these
        moves = takes or moves  # takes have priority
        if moves:
            piece, move = choose_best_move(board=self.board, moves_list=moves)
        else:
            print("SKYNET LOST THE GAME. YOU'VE STOPPED THE MACHINE APOCALYPSE")
            return False

        self.picked_by_pc = piece
        # move = random.choice(moves)
        print(self.get_current_board())
        sleep(1.5)

        self.board[piece] = 1
        bound = self.size
        self.board[move] = -1 * (25 if move[0] + 1 == bound else 5)

        if takes:  # If we captured some piece, we have to delete it
            opponent_piece = tuple(
                map(
                    lambda x: int(x / 2), [
                        (move[0] + piece[0]), (move[1] + piece[1])
                    ]
                )
            )
            self.board[opponent_piece] = 1


        return




        # ______________________________________________________________________________________________


        if takes:  # at least one
            # Then we're obliged to choose one from them
            piece, move = choose_best_move(
                board=self.board, moves_list=takes
            )
        else:
            moves: list[tuple[tuple[int, int]]] = []  # Tuples of start and end position
            [[moves.append((piece, x)) for x in self.get_moves_list(piece, player_pointer=-1, is_king=False)] for piece in pieces]
            if not moves:
                if attempts > self.size:
                    print("SKYNET LOST THE GAME")
                    return
                else:
                    return self.make_pc_move(attempts=attempts + 1)

            piece, move = choose_best_move(board=self.board, moves_list=moves)
        print("FINALLY, ", piece, move)

        # --------------------------------------------------------------------

        self.picked_by_pc = piece
        # move = random.choice(moves)
        print(self.get_current_board())
        # print("MOVE IS ", move)
        sleep(1.5)

        if takes:
            enemy_piece = tuple(map(lambda x: int(x / 2), [(move[0] + piece[0]), (move[1] + piece[1])]))

            # self.human_pieces.remove(
            #     enemy_piece
            # )

            self.board[enemy_piece] = 1

        self.board[piece] = 1
        self.board[move] = -5




        # takes = list(self.get_captures_list(piece, player_pointer=-1))
        # moves = list(self.get_moves_list(piece, player_pointer=-1))
        # print(f"TAKES: \n{takes}\nMOVES: \n{moves}")
        # moves = takes or moves


        # bound = self.size


        # While taking the enemy's piece, we can move to both top and bottom
        # takes = get_moves(piece=piece, step=-2)

        # takes = [
        #     take for take in takes if (0 <= take[0] < bound) and (0 <= take[1] < bound) and all(
        #         (
        #             self.board[take] == 1,
        #             self.board[
        #                 (
        #                     int((take[0] + piece[0]) / 2), int((take[1] + piece[1]) / 2)
        #                 )
        #             ] in (5, 25))
        #     )]


        # If it's no way to take, just check other moves
        # if not takes:
        #     # hypothetical_moves = [
        #     #     (piece[0]-1, piece[1]-1), (piece[0]-1, piece[1]+1)
        #     # ]
        #     hypothetical_moves = get_moves(piece=piece, step=-1, backward=False)
        #
        #     real_moves = [
        #         move for move in hypothetical_moves if 0 <= move[0] < bound and 0 <= move[1] < bound and self.board[move] == 1
        #     ]

        # moves = takes or real_moves

        # if moves:  # At least one way to move
        #
        #     self.picked_by_pc = piece
        #     move = random.choice(moves)
        #     print(self.get_current_board())
        #     # print("MOVE IS ", move)
        #     sleep(1.5)
        #
        #     if takes:
        #         enemy_piece = tuple(map(lambda x: int(x / 2), [(move[0] + piece[0]), (move[1] + piece[1])]))
        #
        #         # self.human_pieces.remove(
        #         #     enemy_piece
        #         # )
        #
        #         self.board[enemy_piece] = 1
        #
        #     self.board[piece] = 1
        #     self.board[move] = -5
        #     # self.computer_pieces.remove(piece)
        #     # self.computer_pieces.append(move)
        # else:
        #     # If system couldn't get at least one move, we'll give to it another
        #     # chance. We will count number of attempts. If PC tried to pick piece
        #     # several times already and each time it failed, we'll say system lost
        #     # the game
        #     if attempts > self.size:
        #         print("SKYNET LOST THE GAME")
        #     else:
        #         return self.make_pc_move(attempts=attempts + 1)


    def make_human_move(self) -> None:
        """
        Function to make move as a human.
        Firstly, player input the coordinates of desired piece.
        By next step we got the square to move it on.
        When something's going wrong with the input, we call this function
        recursively to give to player another chance.
        """
        if not self.computer_pieces:
            return print("YOU LOST THE GAME")
        piece = input("Enter the coordinates of the piece you want to move (row, col): ")
        try:
            piece = tuple(map(int, piece.split(',')))
            assert piece in get_pieces_indexes(board=self.board, rank=5), ValueError
            # assert piece in self.human_pieces, ValueError
        except (ValueError, AssertionError):
            print('You did wrong input. Try again! ')
            return self.make_human_move()

        bound = self.size



        # Firstly, we check if player had to beat enemy piece somewhere.
        # In that case it's not allowed to make other moves

        # takes = [(piece[0] + 2, piece[1] - 2), (piece[0] + 2, piece[1] + 2)]
        # print(takes)

        takes =  get_moves(piece=piece, step=2)

        takes = [
            take for take in takes if (
                0 <= take[0] < bound  # We can't check all condition in one "all"
                ) and (
                0 <= take[1] < bound
                ) and all(
                    (
                        self.board[
                            (
                                int((take[0] + piece[0]) / 2), int((take[1] + piece[1]) / 2)
                            )
                        ] in (-5, -25),
                        self.board[take] == 1,
                    )
                )
        ]
        # print("WAYS TO TAKE: ", takes)

        # If it's no way to take, just check other moves
        if not takes:
            hypothetical_moves = [
                (piece[0] + 1, piece[1] - 1), (piece[0] + 1, piece[1] + 1)
            ]
            hypothetical_moves = get_moves(piece=piece, step=1, backward=False)

            real_moves = [
                move for move in hypothetical_moves if (
                    0 <= move[0] < bound) and (
                     0 <= move[1] < bound) and (
                     self.board[move] == 1
                )
            ]


            # real_moves = [
            #     move for move in hypothetical_moves if 0 <= move[0] < 6 and 0 <= move[1] < 6 and self.board[move] == 1
            # ]
            print("YOUR REAL MOVES: ", real_moves)

        moves = takes or real_moves
        if moves:
            self.picked_by_human = piece
            print(self.get_current_board())
            move = input("Enter the coordinates of the square you want to move to (row, col): ")
            try:
                move = tuple(map(int, move.split(',')))
                assert move in moves or move in takes
            except:
                print('You did wrong input. Try again! ')
                return self.make_human_move()

            self.board[piece] = 1
            self.board[move] = (25 if move[0] + 1 == bound else 5)
            # self.human_pieces.remove(piece)
            # self.human_pieces.append(move)
            if move in takes:
                enemy_piece = tuple(map(lambda x: int(x / 2), [(move[0] + piece[0]), (move[1] + piece[1])]))
                # self.computer_pieces.remove(
                #     enemy_piece
                # )
                self.board[enemy_piece] = 1
        else:
            print("NO WAY TO MOVE THIS PIECE! Try again! ")
            return self.make_human_move()


if __name__ == "__main__":
    # chess = Game(size=4, board_type='emoji')
    # print(chess.get_current_board())
    chess = Game(size=6, board_type='emoji')
    # print(chess.get_current_board())
    # chess = Game(size=8, board_type='emoji')
    # print(chess.get_current_board())
    # chess = Game(size=10, board_type='emoji')
    # print(chess.get_current_board())
    # chess = Game(size=6, board_type='numeric')
    # board = chess.get_current_board()
    # print(get_pieces_indexes(board=board, rank=5))
    # res = np.where(board == 5)
    # print(np.any(board == 5, board == -5))
    # print(res, type(res))
    # print(res[0], res[1])
    # print(list(zip(res[0], res[1])))



# Play the game
    while True:
        print(chess.get_current_board())
        print(chess.board)
        chess.make_human_move()
        print('Computer is making a move...')
        chess.make_pc_move()
