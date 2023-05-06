"""

"""
from aiogram import Router, F, Bot, exceptions
from aiogram.types import Message, CallbackQuery, BotCommand, LabeledPrice, SuccessfulPayment

from aiogram.fsm.context import FSMContext

router = Router()  # [1]
from app.set_menu_commands import set_personal_menu_commands
from keyboards import BasicKeyboards as bkb
from keyboards import DynamicKeyboards as dkb


from text_data.message_answers import answers_texts as ma_texts
from text_data.callback_answers import param_change
from text_data.user_settings import PROPERTIES_DESCRIPTIONS as property_info


@router.callback_query(F.data == 'settings')
async def callbacks_num_change_fab(
        callback: CallbackQuery,
        state: FSMContext
):
    info = await state.get_data()
    lang = info['language']
    # await callback.message.edit_reply_markup()
    try:
        await callback.message.edit_text(
            ma_texts['settings'][lang] + ma_texts['settings_main'][lang],
            reply_markup=await dkb.show_all_properties(info=info, lang=lang))
    except UnboundLocalError:
        # we lost data after restarting the bot
        await callback.message.answer(ma_texts['main'][lang], reply_markup=bkb.main_page_kb[lang])



@router.callback_query(dkb.PropertiesCallback.filter())
async def callbacks_num_change_fab(
        callback: CallbackQuery,
        callback_data: dkb.PropertiesCallback,
        state: FSMContext,
        bot: Bot
):
    lang = (await state.get_data())['language']
    param = callback_data.param
    action = callback_data.action
    value = callback_data.value
    data = callback_data.data

    if action == 'show':  # Показать текущее значение или доступные варианты
        # markup = await dkb.get_property_status(param=param, value=value, lang=lang)
        # new_value = None

        await callback.message.edit_text(
            ma_texts['settings'][lang] + property_info[param][lang],
            reply_markup=await dkb.get_property_status(param=param, value=value, lang=lang))
        return

    elif action =='set':
        # value = callback_data.data
        # print('WE try to set ', value)
        new_value = value
        if param == 'language':
            lang = data
            new_value = data
            user_id = callback.from_user.id
            chat_id = callback.message.chat.id
            await set_personal_menu_commands(
                chat_id=chat_id, user_id=user_id, lang=lang, bot=bot
            )
            await callback.message.edit_text(
                ma_texts['settings'][lang] + property_info[param][lang], reply_markup=None
            )

        # markup = await DK.show_properties(lang=lang)
        markup = await dkb.get_property_status(param=param, value=new_value, lang=lang)
    elif action == 'switch':
        # Делаем Булеву инверсию
        new_value = not value
        markup = await dkb.get_property_status(param=param, value=new_value, lang=lang)

    try:
        await callback.message.edit_reply_markup()
    except exceptions.TelegramBadRequest:
        # The only reason for this exception is case we change the language.
        # Just pass it because we've already deleted the markup
        pass

    kwargs = {param: new_value}
    await state.update_data(**kwargs)
    # print(kwargs)
    await callback.answer(param_change[lang])
    # await callback.message.edit_text(
    #     ma_texts['settings'][lang] + property_info[param][lang], reply_markup=markup
    # )
    await callback.message.edit_reply_markup(reply_markup=markup)
