import asyncio

from aiogram import Router, F, Bot
from aiogram.types import Message, CallbackQuery
from aiogram.fsm.context import FSMContext

router = Router()  # [1]

from keyboards import CasinoGame

from processing.casinoBackEnd import get_slot_machine_result

from text_data.message_answers import answers_texts as ma_texts
from text_data.callback_answers import param_change
from processing.SQL_processingg import SQL_high_level_processing as sql_high_p


@router.callback_query(F.data == 'casino')
async def callbacks_num_change_fab(
        callback: CallbackQuery,
        state: FSMContext
):
    lang = (await state.get_data())['language']
    # print(callback.game_short_name)
    await callback.message.edit_text(ma_texts['slot_machine']['main'][lang], reply_markup=CasinoGame.casino_start_kb[lang])


@router.callback_query(CasinoGame.CasinoCallback.filter())
async def callbacks_num_change_fab(
        callback: CallbackQuery,
        callback_data: CasinoGame.CasinoCallback,
        state: FSMContext
):
    id = callback.from_user.id
    lang = (await state.get_data())['language']
    try:
        bet = (await state.get_data())['sm_bet']
    except KeyError:
        bet = 10
        await state.update_data(sm_bet=bet)
    action = callback_data.action
    if action == 'play':
        await callback.message.edit_reply_markup()
        dice = await callback.message.answer_dice(emoji='🎰')
        result, coins = get_slot_machine_result(
            value=dice.dice.value,
            bet=bet,
            lang=lang
        )
        await sql_high_p.change_coins_balance(id, charge=coins, method='put')
        await asyncio.sleep(1.5)

        await dice.answer(result, reply_markup=CasinoGame.casino_play_kb[lang])

        # await dice.answer("VALUE IS:    "+str(dice.dice.value), reply_markup=CasinoGame.casino_play_kb[lang])
    elif action == 'show_settings':
        await callback.message.edit_text(ma_texts['slot_machine']['bets'][lang], reply_markup=await CasinoGame.get_bet_settings(value=bet, lang=lang))

    elif action == 'set_bet':
        value = callback_data.value

        await callback.message.edit_reply_markup()
        await state.update_data(sm_bet=value)
        await callback.answer(param_change[lang])
        await callback.message.edit_reply_markup(
                                         reply_markup=await CasinoGame.get_bet_settings(value=value, lang=lang))


    elif action == 'show_rules':
        await callback.message.edit_text(ma_texts['slot_machine']['rules'][lang], reply_markup=CasinoGame.casino_help_kb[lang])

