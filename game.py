import numpy as np
import random
from time import sleep

# Initialize the board
# board = np.zeros((6, 6), dtype=str)
# board = np.zeros((6, 6), dtype=int)
# board = np.arange((6, 6), dtype=int)
board = np.full((6, 6), fill_value='⬜', dtype=str)
'⚫⚪⬛⬜'
board[1::2, ::2] = '⬛'
board[::2, 1::2] = '⬛'


# Initialize the pieces
pieces = {'h': [(0, 1), (0, 3), (0, 5), (1, 0), (1, 2), (1, 4)],
          'c': [(4, 1), (4, 3), (4, 5), (5, 0), (5, 2), (5, 4)]}

# Print the board
def print_board(board, pieces, current_piece: tuple | None = None):
    print('______', end='')
    for j in range(len(board[0])):
        print(j, end=')____')

    print('')
    for i in range(len(board)):
        print(i, end=')    ')

        for j in range(len(board[i])):
            if (i, j) in pieces['h']:
                if (i, j) == current_piece:
                    board[i][j] = '🟣'
                else:
                    board[i][j] = '⚫'
            elif (i, j) in pieces['c']:
                if (i, j) == current_piece:
                    board[i][j] = '🟠'
                else:
                    board[i][j] = '🟡'
            print(board[i][j], end='__|')
        print()

# Make a move for the computer
def computer_move(board, pieces):
    piece = random.choice(list(pieces['c']))
    # print(piece)
    moves = [(piece[0]-1, piece[1]-1), (piece[0]-1, piece[1]+1)]
    moves = [move for move in moves if 0 <= move[0] < 6 and 0 <= move[1] < 6 and board[move] == '⬛']
    # print(moves)
    if moves:
        move = random.choice(moves)
        print_board(board, pieces, current_piece=piece)
        sleep(3)
        board[piece] = '⬛'
        board[move] = '🟡'
        pieces['c'].remove(piece)
        pieces['c'].append(move)

# Make a move for the human player
def human_move(board, pieces):
    piece = input("Enter the coordinates of the piece you want to move (row, col): ")
    piece = tuple(map(int, piece.split(',')))
    moves = [(piece[0]+1, piece[1]-1), (piece[0]+1, piece[1]+1)]
    # print(moves)
    short_moves = [move for move in moves if 0 <= move[0] < 6 and 0 <= move[1] < 6 and board[move] == '⬛']
    takes = [(piece[0] + 2, piece[1] - 2), (piece[0] + 2, piece[1] + 2)]
    print(takes)
    print('START: {}\nENEMY: {}\nEND: {}'.format(piece, f'{tuple(map(int, ((takes[0][0] + piece[0])/2, (takes[0][1] + piece[1])/2)))}', takes[0]))
    takes = [
        take for take in takes if (0 <= take[0] < 6) and (0 <= take[1] < 6) and all(
            (
             board[take] == '⬛',
             board[
                (
                    int((take[0] + piece[0])/2), int((take[1] + piece[1])/2)
                 )
            ] == '🟡')
        )]
    moves = [*short_moves, *takes]
    print(moves)
    if moves:

        print_board(board, pieces, current_piece = piece)
        move = input("Enter the coordinates of the square you want to move to (row, col): ")
        move = tuple(map(int, move.split(',')))
        if move in moves:
            board[piece] = '⬛'
            board[move] = '⚫'
            pieces['h'].remove(piece)
            pieces['h'].append(move)
            if move in takes:
                enemy_piece = tuple(map(lambda x: int(x/2), [(move[0] + piece[0]), (move[1] + piece[1])] ))
                pieces['c'].remove(
                    enemy_piece
                )
                board[enemy_piece] = '⬛'
        else:
            print('unacceptable move! '.upper())


# Play the game
while True:
    print_board(board, pieces)
    human_move(board, pieces)
    print('Computer is making a move...')
    computer_move(board, pieces)