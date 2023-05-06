import asyncio

from aiogram import Router, F, Bot, exceptions
# from aiogram.filters import or_f, in_
from aiogram.types import Message, CallbackQuery, BotCommand, LabeledPrice, SuccessfulPayment, PreCheckoutQuery, successful_payment, ContentType
from aiogram.filters import CommandStart, Command, Text, Filter, CommandObject
from aiogram.methods.send_message import SendMessage
from aiogram.fsm.context import FSMContext
from aiogram.utils.deep_linking import create_start_link

from secret_data import TG_ADMIN_ID, TG_SUPPORT_ID
router = Router()  # [1]
from app.finite_state_machine import UserStates
from keyboards import BasicKeyboards as bkb
from keyboards import DynamicKeyboards as dkb
from keyboards.CommunicationWithAdmin import FeedbackCallback
from app.set_menu_commands import set_personal_menu_commands


from processing.SQL_processingg.SQL_high_level_processing import get_lang, get_info_by_nickname, get_user_info_quickly
from processing.AccountProcessing import get_account_details
# from processing.casinoBackEnd import get_slot_machine_resul

from text_data.message_answers import answers_texts as ma_texts

from text_data import callback_answers
from payments.pay_text_data import main_market_description
from payments.pay_keyboards import market_kb, first_market_opening_kb


@router.callback_query(F.data == 'main')
async def callbacks_num_change_fab(
        callback: CallbackQuery,
        state: FSMContext
):
    # lang = await sqlp.get_lang(id)
    lang = (await state.get_data())['language']
    # user_id = callback.from_user.id
    # chat_id = callback.message.chat.id
    # await set_personal_menu_commands(
    #     chat_id=chat_id, user_id=user_id, lang=lang, bot=bot
    # )

    # await callback.message.delete()
    # await callback.message.answer('MAIN PAGE', reply_markup=bkb.main_page_kb[lang])
    await callback.message.edit_text(ma_texts['main'][lang], reply_markup=bkb.main_page_kb[lang])
    # await callback.message.edit_reply_markup(reply_markup=bkb.main_page_kb[lang])


@router.callback_query(F.data == 'help')
async def callbacks_num_change_fab(
        callback: CallbackQuery,
        state: FSMContext
):
    lang = (await state.get_data())['language']
    # await callback.message.edit_reply_markup()
    await callback.message.edit_text(ma_texts['help'][lang], reply_markup=bkb.help_kb[lang])

from aiogram.types.labeled_price import LabeledPrice
price = LabeledPrice(label='СЛОН', amount=10000)
prices = [price]


@router.callback_query(F.data == 'account')
async def account_command_handler(callback: CallbackQuery, state: FSMContext) -> None:
    """
    Command to see information about yourself
    """
    id = callback.from_user.id
    lang = (await state.get_data())['language']
    await callback.message.edit_text(text=await get_account_details(
        callback.from_user.id, lang
    ), reply_markup=bkb.account_from_menu_kb[lang])
    # # lang = 'ru'

    # info = await get_user_info_quickly(id)
    # info['balance'] = md.quote(str(info['balance']))  # in case it's negative
    # expiry = await timep.get_expiry_date(info['expiry'])
    # info['expiry'] = expiry.strftime(expiry_format['coins_update'][lang])
    # info['status'] = info['status'].upper()
    # await callback.message.edit_text(ma_texts['account'][lang].format(**info), reply_markup=bkb.back_to_main_kb[lang])


@router.callback_query(F.data == 'market')
async def send_invoice(
        callback: CallbackQuery,
        state: FSMContext,
        bot: Bot
):
    lang = (await state.get_data())['language']
    # 381764678: TEST:51844
    await callback.message.edit_text(
        ma_texts['market'][lang]+main_market_description[lang],
        reply_markup=first_market_opening_kb[lang])
    # await callback.message.reply_invoice(
    #     title='КУПИ',
    #     description='купи',
    #     payload='покупка',
    #     provider_token='381764678: TEST:51844',
    #     currency='RUB',
    #     prices=prices,
    #     allow_sending_without_reply = True
    #
    # )
    #
    # await callback.message.reply('ВОТ', reply_markup=bkb.tinkoff)

@router.callback_query(F.data == 'cancel')
async def callbacks_num_change_fab(
        callback: CallbackQuery,
        state: FSMContext
):
    lang = (await state.get_data())['language']
    await state.set_state(UserStates.main)
    await callback.answer(callback_answers.forget_action[lang])
    await callback.message.delete()
# await callback.message.edit_reply_markup(reply_markup=bkb.setting_kb[lang])

@router.callback_query(F.data == 'restriction_reason')
async def send_invoice(
        callback: CallbackQuery,
        state: FSMContext,
        bot: Bot
):
    lang = (await state.get_data())['language']
    # 381764678: TEST:51844
    await callback.answer(callback_answers.restriction_reason_in_help[lang], show_alert=True)

@router.callback_query(FeedbackCallback.filter())
async def confirm_feedback(
        callback: CallbackQuery,
        callback_data: FeedbackCallback,
        state: FSMContext
):
    id = callback.from_user.id

    state_data = (await state.get_data())
    lang = state_data['language']
    if callback_data.action == 'send_to_admin':
        info = await get_user_info_quickly(id)
        username = info['nickname']
        status = info['status']
        referrals = info['referrals']
        balance = info['balance']
        await SendMessage(
            chat_id=TG_SUPPORT_ID,
            text = ma_texts['feedback']['message_from_user'].format(
                username, id, status, balance, referrals
            ),
            # parse_mode='MarkdownV2'
        )
        await callback.message.forward(chat_id=TG_SUPPORT_ID,)
        await callback.message.delete_reply_markup()
        await callback.message.reply(ma_texts['feedback']['was_sent_to_admin'][lang])

    elif callback_data.action == 'send_to_user':
        user_to_send = state_data['user2contact']

        try:
            # Попробуем превратить в INTEGER, если у нас ID
            # arg = {'id': int(user_to_send)}
            user_id = int(user_to_send)
        except ValueError:  # Ловим ошибку, если это никнейм
            user_id = await get_info_by_nickname(user_to_send[1:])  # Срезаем @
            # arg = {'nickname': user_to_send[1:]}  # Срезаем @
        lang = await get_lang(user_id)
        if lang:
            await SendMessage(
                chat_id=user_id,
                text = ma_texts['feedback']['message_from_admin'][lang].format(
                    callback.message.text
                ),
            )
            # await callback.message.forward(chat_id=TG_SUPPORT_ID,)
            await callback.message.delete_reply_markup()
            await state.set_state(UserStates.main)
            await callback.message.reply(ma_texts['feedback']['was_sent_to_user'])
        else:
            await callback.message.reply("ERROR")

    elif callback_data.action == 'cancel':
        await callback.answer(callback_answers.forget_action[lang])
        await callback.message.delete()
    # await callback.message.edit_reply_markup()
    # await callback.message.edit_text(ma_texts['help'][lang], reply_markup=bkb.cancel_kb[lang])


@router.callback_query(F.data == 'referral')
async def callbacks_num_change_fab(
        callback: CallbackQuery, bot: Bot,
        state: FSMContext
):
    """ Функция для генерации и отправки реферальной ссылки """
    # await callback.message.edit_reply_markup()
    lang = (await state.get_data())['language']
    link = await create_start_link(bot=bot,
        payload = str(callback.from_user.id),
        encode=True
    )
    await callback.message.edit_text(ma_texts['referral']['message'][lang].format(link), reply_markup=await dkb.get_fererral_kb(
        lang=lang, link=ma_texts['referral']['referral_text'][lang].format(link)))


# @router.callback_query(F.data == 'ready')
@router.callback_query(F.data.in_({'ready', }), UserStates.need_to_unblock_bot)
async def callbacks_num_change_fab(
        callback: CallbackQuery, bot: Bot,
        state: FSMContext
):
    user_id = callback.from_user.id
    chat_id = callback.message.chat.id
    lang = (await state.get_data())['language']
    # user_id = callback.from_user.id
    # chat_id = callback.message.chat.id
    # await set_personal_menu_commands(
    #     chat_id=chat_id, user_id=user_id, lang=lang, bot=bot
    # )
    # await state.update_data(language=lang)
    await callback.answer(callback_answers.language_set[lang])
    if chat_id == user_id:
        await callback.message.edit_text(
            ma_texts['start']['private'][lang]
        )
        # await sql_high_p.add_new_user(user_id, username, lang, now)
    else:
        try:
            await SendMessage(
                chat_id=user_id,
                text=ma_texts['start']['private'][lang]
            )
            await callback.message.edit_text(
                ma_texts['start']['group'][lang]
            )
        except exceptions.TelegramForbiddenError:  # Bot Blocked
            await callback.message.edit_text(
                ma_texts['start']['failed'][lang]
            )
            return
    await state.set_state(UserStates.main)
    await set_personal_menu_commands(
        chat_id=chat_id, user_id=user_id, lang=lang, bot=bot
    )



# @router.callback_query(F.data.in_({'en', 'ru'}), UserStates.main)
# async def callbacks_num_change_fab(
#         callback: CallbackQuery, bot: Bot,
#         state: FSMContext
# ):
#     user_id = callback.from_user.id
#     chat_id = callback.message.chat.id
#     print(user_id, chat_id)
#     lang = callback.data
#     await state.update_data(language=lang)
#     await callback.answer(callback_answers.language_set[lang])
#     if chat_id == user_id:
#         await callback.message.edit_text(
#             ma_texts['start']['private'][lang]
#         )
#         # await sql_high_p.add_new_user(user_id, username, lang, now)
#     else:
#         try:
#             await SendMessage(
#                 chat_id=user_id,
#                 text=ma_texts['start']['private'][lang]
#             )
#             await callback.message.edit_text(
#                 ma_texts['start']['group'][lang]
#             )
#         except exceptions.TelegramForbiddenError:  # Bot Blocked
#             await callback.message.edit_text(
#                 ma_texts['start']['failed'][lang]
#             )
#             # await state.clear()