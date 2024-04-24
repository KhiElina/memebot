# Установка всех необходимых для работы нашей программы модулей(библиотек)
!pip install pytelegrambotapi -q
!pip install g4f -q
!pip install diffusers -q
!pip install deep-translator -q


import telebot;# Подключаем бота
bot = telebot.TeleBot('6617358370:AAHaGEknTvx1MgM1zZsWXmoVucadPN7XdxQ');# Вводим наш токен(бота)

import g4f # Подключаем материалы для работы GPT4 (for free)
from g4f.Provider import Bing, OpenaiChat, Liaobots, BaseProvider # В случае ошибки связанной с неккоректными данными выбираем любой из этих провайдеров и меняем и в коде ниже(укажу где)
from g4f.cookies import set_cookies
from g4f.client import Client

from deep_translator import GoogleTranslator # Подключаем гугл-переводчик

import nest_asyncio

from diffusers import DiffusionPipeline # Подключаем Диффузоры(библиотека питона для создания изображений)
import torch



nest_asyncio.apply()
client= Client()
chat_history = [{"role": "user", "content": ''}] # Создание истории диалога для GPT4 (for free)

# Функция для работы с GPT4 (for free) (её дал старшеклассник Акимова, за код никто не шарит)
def send_request(message):
    global chat_history
    chat_history[0]["content"] += message + " "
    try:
        answer = g4f.ChatCompletion.create(
        model=g4f.models.default,
        provider=g4f.Provider.Liaobots, # Меняем здесь последнюю часть(то что после 2ой точки)
        messages=chat_history
    )
    except Exception as err:
        answer = g4f.ChatCompletion.create(
        model=g4f.models.default,
        provider=g4f.Provider.Liaobots, # Меняем здесь последнюю часть(то что после 2ой точки)
        messages=chat_history
    )
    return (answer) # Возвращаем ответ от нейронки (в моем случае шутку)
    chat_history[0]["content"] += answer + " "

def send_photo1(message): # Функция принимает значение текстового формата
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
    return image # Возвращаем картинку


# Приветствие при первом запуске бота
@bot.message_handler(commands=['start'])
def start(message):
    bot.send_message(message.chat.id, 'Привет! Можешь начинать! Время шуток!')


# Если сообщение текстовое то,(иначе выдаст обычную ошибку)
@bot.message_handler(content_types=['text'])  # Проверка формата сообщения
def get_text_messages(message):  # Берем полученное сообщение в боте
    input_1 = GoogleTranslator(source='auto', target='en').translate(
        message.text)  # Наше сообщение переводим в английский
    input_to_ai = "Сome up with a joke about " + input_1  # Помимо нашего сообщения, просим его пошутить на "Наше сообщение"
    output = GoogleTranslator(source='auto', target='ru').translate(
        send_request(input_to_ai))  # Полученный ответ переводим на русский

    # Отправляем сообщение
    bot.send_message(message.chat.id, output)
    # Отправляем картинку созданной по нашей шутке
    bot.send_photo(message.chat.id, send_photo1(output))  # Запрашиваем картинку по нашей фотке


bot.polling(none_stop=True, interval=0)  # Зацикливаем работу бота
