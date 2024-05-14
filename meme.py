# Установка всех необходимых для работы нашей программы модулей(библиотек)
!pip install pytelegrambotapi - q
!pip install - U g4f - q
!pip install browser - cookie3 - q
!pip install aiohttp_socks - q
!pip install diffusers - q
!pip install deep - translator - q

# Подключаем бота
import telebot;

bot = telebot.TeleBot('6617358370:AAHaGEknTvx1MgM1zZsWXmoVucadPN7XdxQ');  # Вводим наш токен(бота)
from deep_translator import GoogleTranslator  # Подключаем гугл-переводчик
from diffusers import DiffusionPipeline  # Подключаем Диффузоры(библиотека питона для создания изображений)
import torch
from telebot import types

import g4f
from g4f.Provider import (
    GeekGpt,
    Liaobots,
    Phind,
    Raycast,
    RetryProvider)
from g4f.client import Client
import nest_asyncio

nest_asyncio.apply()

client = Client(
    provider=RetryProvider([
        g4f.Provider.Liaobots,
        g4f.Provider.GeekGpt,
        g4f.Provider.Phind,
        g4f.Provider.Raycast
    ])
)
chat_history = [{"role": "user", "content": 'Отвечай на русском языке'}]


def send_request(message):
    global chat_history
    chat_history[0]["content"] += message + " "

    try:
        response = g4f.ChatCompletion.create(
            model=g4f.models.gpt_4,
            messages=chat_history
        )
    except Exception as err:
        print("Все провайдеры не отвечают, попробуйте пойзже")
    print(chat_history)
    chat_history[0]["content"] += response + " "
    return response


def send_photo1(message):  # Функция принимает значение текстового формата
    base = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True)

    base.to("cuda")
    refiner = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        text_encoder_2=base.text_encoder_2,
        vae=base.vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16", )
    refiner.to("cuda")

    n_steps = 40
    high_noise_frac = 0.8

    prompt = message

    image = base(
        prompt=prompt,
        num_inference_steps=n_steps,
        denoising_end=high_noise_frac,
        output_type="latent", ).images
    image = refiner(
        prompt=prompt,
        num_inference_steps=n_steps,
        denoising_start=high_noise_frac,
        image=image, ).images[0]
    return image  # Возвращаем картинку


@bot.message_handler(commands=['start'])
def start(message):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    btn1 = types.KeyboardButton("👋 Поздороваться 👋")
    btn2 = types.KeyboardButton("❓ Задать вопрос ❓")
    markup.add(btn1, btn2)
    bot.send_message(message.chat.id, 'Привет! Привет! Что тебя интересует?', reply_markup=markup)


@bot.message_handler(content_types=['text'])
def func(message):
    if (message.text == "👋 Поздороваться 👋"):
        bot.send_message(message.chat.id, text="Привет! Привет! Спасибо что пользуешься мной!!!")
    elif (message.text == "❓ Задать вопрос ❓"):
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        btn1 = types.KeyboardButton("Как тебя зовут?")
        btn2 = types.KeyboardButton("Для чего я создан?")
        back = types.KeyboardButton("Вернуться на главную")
        markup.add(btn1, btn2, back)
        bot.send_message(message.chat.id, text="Задай мне вопрос", reply_markup=markup)

    elif (message.text == "Как тебя зовут?"):
        bot.send_message(message.chat.id, "Meme Generator Bot")

    elif message.text == "Для чего я создан?":
        bot.send_message(message.chat.id, text=" Придумать шутку и изобразить её, внезависимисти от твоего предложения")

    elif (message.text == "Вернуться на главную"):
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        button1 = types.KeyboardButton("👋 Поздороваться 👋")
        button2 = types.KeyboardButton("❓ Задать вопрос ❓")
        markup.add(button1, button2)
        bot.send_message(message.chat.id, text=" Вы вернулись на главную", reply_markup=markup)
    else:
        input_1 = GoogleTranslator(source='auto', target='en').translate(
            message.text)  # Наше сообщение переводим в английский
        input_to_ai = "Сome up with a joke about " + input_1  # Помимо нашего сообщения, просим его пошутить на "Наше сообщение"
        output = GoogleTranslator(source='auto', target='ru').translate(
            send_request(input_to_ai))  # Полученный ответ переводим на русский
        # Отправляем сообщение (супер мега)(шутка от бота)
        bot.send_message(message.chat.id, output)
        # Отправляем (супер мега) картинку созданной по нашей шутке
        bot.send_photo(message.chat.id, send_photo1(output))  # Запрашиваем картинку по нашей фотке


bot.polling(none_stop=True, interval=0)