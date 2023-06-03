"""

"""

from aiogram import Router, F, md, html, Bot
from aiogram.fsm.context import FSMContext
from aiogram.types import Message, CallbackQuery
from aiogram.filters import CommandStart, Command, Text, Filter, CommandObject, invert_f, or_f, and_f

router = Router()
from aiogram.types import InlineKeyboardMarkup
from aiogram.utils.keyboard import InlineKeyboardBuilder

from aiogram.types import (ReplyKeyboardRemove, ReplyKeyboardMarkup,
                           KeyboardButton, InlineKeyboardMarkup, InlineKeyboardButton)

import numpy as np
arr = np.zeros((4, 5), dtype=int)

# Making a chessboard for out game
CHESSBOARD = np.zeros((4, 5), dtype=int)
CHESSBOARD[::2, ::2] = 1
CHESSBOARD[1::2, 1::2] = 1
print("CURRENT BOARD:\n", CHESSBOARD)
# Calculate squares on which we can move
AVAILABLE_SQUARES = np.count_nonzero(CHESSBOARD)
# print(arr)
# CHESSBOARD = np.array([
#     [1, 0, 1, 0, 1],
#     [0, 1, 0, 1, 0],
#     [1, 0, 1, 0, 1],
#     [0, 1, 0, 1, 0],
# ])
CHESSBOARD[0, 0] = 10
CHESSBOARD[0, 2] = 100
CHESSBOARD[0, 4] = 10
print(CHESSBOARD)


def build_chessboard():
    builder = InlineKeyboardBuilder()
    shape = CHESSBOARD.shape
    for row in CHESSBOARD:
        for cell in row:

            if cell == 0:
                builder.button(
                    text="     ", callback_data='data'
                )
            elif cell == 1:
                builder.button(
                    text="⬜", callback_data='data'
                )
            elif cell == 10:
                builder.button(
                    text="🔴", callback_data='data'
                )
            elif cell == 100:
                builder.button(
                    text="🟢", callback_data='data'
                )
        #     print(cell, end=';  ')
        # print('\n')
    builder.adjust(shape[1], repeat=True)
    # builder.adjust(list([5 for _ in range(4)]))
    # builder.adjust(list(np.full(shape[0], shape[1]).tolist()))
    # builder.adjust(list([int(i) for i in np.full(4, 5).tolist()]))
    return builder.as_markup()


@router.message(CommandStart())
async def command_start_handler(message: Message, command: CommandObject, state: FSMContext) -> None:
    """

    """
    user_id, chat_id, name, username = (
        message.from_user.id, message.chat.id, message.from_user.first_name,
        message.from_user.username
    )
    await message.reply('ВОТ', reply_markup=build_chessboard())
