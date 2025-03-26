import asyncio
import os
import cv2
import yaml
import logging
import importlib.util
import time
from aiogram import Bot, Dispatcher, types
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton, ContentType, FSInputFile
from aiogram.enums import ParseMode
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.filters import Command, StateFilter
from aiogram.client.default import DefaultBotProperties
from utils.yolo_utils import recognize_faces, load_known_faces, save_known_face

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Загрузка конфигурации из YAML файла
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Инициализация бота и диспетчера
bot_token = config["telegram"]["bot_token"]
chat_id = config["telegram"]["chat_id"]
admin_ids = config["telegram"]["admin_ids"]
bot = Bot(token=bot_token, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
dp = Dispatcher(storage=MemoryStorage())

# Проверка доступности CUDA
cuda_available = cv2.cuda.getCudaEnabledDeviceCount() > 0
cuda_warning_shown = False

if cuda_available:
    logger.info("CUDA доступен, используется GPU")
else:
    logger.warning("CUDA недоступна, используется CPU")

# Путь для хранения известных лиц
known_faces_path = config["known_faces_path"]
os.makedirs(known_faces_path, exist_ok=True)

# Загрузка известных лиц
known_face_encodings, known_face_names = load_known_faces(known_faces_path)

# Словарь для хранения пути к последнему кадру для каждого потока
last_frames_paths = {stream: None for stream in config["streams"].keys()}

# Словарь для отслеживания обнаруженных лиц (поток: множество имён)
detected_faces = {stream: set() for stream in config["streams"].keys()}

# Словарь для отслеживания времени последнего обнаружения лица (поток: {имя: время})
last_seen_time = {stream: {} for stream in config["streams"].keys()}

# Словарь для запросов на получение кадров (поток: chat_id)
frame_requests = {}

# Время для сброса состояния (3 секунды)
RESET_TIMEOUT = 3

# Загрузка плагинов
plugins_dir = "plugins"
os.makedirs(plugins_dir, exist_ok=True)
plugins = {}

for filename in os.listdir(plugins_dir):
    if filename.endswith(".py") and not filename.startswith("__"):
        plugin_name = filename[:-3]  # Убираем .py
        spec = importlib.util.spec_from_file_location(plugin_name, os.path.join(plugins_dir, filename))
        plugin_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(plugin_module)
        if hasattr(plugin_module, "setup"):
            plugins[plugin_name] = plugin_module
            logger.info(f"Загружен плагин: {plugin_name}")

# Состояния для FSM
class AddFace(StatesGroup):
    waiting_for_photo = State()
    waiting_for_name = State()

class StreamSelection(StatesGroup):
    waiting_for_stream = State()

# Функция для создания главного меню с учётом прав
def create_main_menu(is_admin=False):
    keyboard = [
        [KeyboardButton(text="Начать")],
    ]
    if is_admin:
        keyboard.extend([
            [KeyboardButton(text="Добавить лицо")],
            [KeyboardButton(text="Получить кадр")],
        ])
        for plugin_name in plugins.keys():
            keyboard.append([KeyboardButton(text=f"Плагин: {plugin_name}")])
    return ReplyKeyboardMarkup(keyboard=keyboard, resize_keyboard=True)

# Функция для создания клавиатуры выбора потоков
def create_streams_keyboard():
    keyboard = [[KeyboardButton(text=str(stream))] for stream in config["streams"].keys()]
    keyboard.append([KeyboardButton(text="Отмена")])
    return ReplyKeyboardMarkup(keyboard=keyboard, resize_keyboard=True)

# Проверка, является ли пользователь администратором
def is_admin(user_id):
    return user_id in admin_ids

# Обработка команды /start
@dp.message(Command(commands=["start"]))
async def start(message: types.Message, state: FSMContext):
    user_id = message.from_user.id
    main_menu_reply_markup = create_main_menu(is_admin(user_id))
    await state.clear()  # Очищаем состояние при старте
    await message.answer("Добро пожаловать! Обработка потоков уже запущена. Выберите действие:", reply_markup=main_menu_reply_markup)

# Обработка выбора команды через кнопки
@dp.message(lambda message: message.text and (message.text in ["Начать", "Добавить лицо", "Получить кадр"] or message.text.startswith("Плагин: ")))
async def handle_choice(message: types.Message, state: FSMContext):
    user_id = message.from_user.id
    main_menu_reply_markup = create_main_menu(is_admin(user_id))
    
    if not is_admin(user_id) and message.text != "Начать":
        await message.answer("У вас нет прав на выполнение этой команды.", reply_markup=main_menu_reply_markup)
        await state.clear()
        return

    choice = message.text
    if choice == "Начать":
        await message.answer("Обработка видеопотоков уже запущена автоматически.", reply_markup=main_menu_reply_markup)
    elif choice == "Добавить лицо":
        await message.answer("Пожалуйста, отправьте фотографию лица для добавления.", reply_markup=main_menu_reply_markup)
        await state.set_state(AddFace.waiting_for_photo)
    elif choice == "Получить кадр":
        await state.set_state(StreamSelection.waiting_for_stream)
        await message.answer("Выберите поток:", reply_markup=create_streams_keyboard())
    elif choice.startswith("Плагин: "):
        plugin_name = choice[len("Плагин: "):]
        if plugin_name in plugins and hasattr(plugins[plugin_name], "execute"):
            result = await plugins[plugin_name].execute(bot, message, chat_id, detected_faces)
            await message.answer(result, reply_markup=main_menu_reply_markup)
        else:
            await message.answer(f"Плагин {plugin_name} не найден или не поддерживает выполнение.", reply_markup=main_menu_reply_markup)
        await state.clear()

# Обработка получения фотографии для добавления лица
@dp.message(StateFilter(AddFace.waiting_for_photo), lambda message: message.content_type == ContentType.PHOTO)
async def receive_photo(message: types.Message, state: FSMContext):
    user_id = message.from_user.id
    main_menu_reply_markup = create_main_menu(is_admin(user_id))
    
    if not is_admin(user_id):
        await message.answer("У вас нет прав на добавление лиц.", reply_markup=main_menu_reply_markup)
        await state.clear()
        return

    temp_file_path = os.path.join(known_faces_path, f"temp_{user_id}_{message.message_id}.jpg")

    try:
        file_id = message.photo[-1].file_id
        await bot.download(file_id, destination=temp_file_path)
        logger.info(f"Фото сохранено как {temp_file_path}")
        await message.answer(f"Фотография временно сохранена как {temp_file_path}", reply_markup=main_menu_reply_markup)

        new_face_encoding, _ = save_known_face(temp_file_path)
        if new_face_encoding is not None:
            logger.info(f"Лицо успешно распознано в {temp_file_path}")
            await message.answer("Лицо распознано. Укажите имя для этого лица.", reply_markup=main_menu_reply_markup)
            await state.update_data(temp_file_path=temp_file_path)
            await state.set_state(AddFace.waiting_for_name)
        else:
            logger.info(f"Лицо не обнаружено в {temp_file_path}")
            await message.answer("Лицо не обнаружено на фотографии.", reply_markup=main_menu_reply_markup)
            os.remove(temp_file_path)
            await state.clear()
    except Exception as e:
        logger.error(f"Ошибка при обработке фото: {str(e)}")
        await message.answer(f"Ошибка при обработке фото: {str(e)}", reply_markup=main_menu_reply_markup)
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        await state.clear()

# Обработка ввода имени для нового лица
@dp.message(StateFilter(AddFace.waiting_for_name), lambda message: message.text is not None)
async def receive_name(message: types.Message, state: FSMContext):
    user_id = message.from_user.id
    main_menu_reply_markup = create_main_menu(is_admin(user_id))
    
    if not is_admin(user_id):
        await message.answer("У вас нет прав на добавление лиц.", reply_markup=main_menu_reply_markup)
        await state.clear()
        return

    name = message.text.strip()
    data = await state.get_data()
    temp_file_path = data.get("temp_file_path")

    if not temp_file_path or not os.path.exists(temp_file_path):
        logger.error("Временный файл не найден")
        await message.answer("Ошибка: временный файл не найден.", reply_markup=main_menu_reply_markup)
        await state.clear()
        return

    try:
        file_path = os.path.join(known_faces_path, f"{name}.jpg")
        if os.path.exists(file_path):
            os.remove(file_path)
        os.rename(temp_file_path, file_path)
        new_face_encoding, _ = save_known_face(file_path)
        if new_face_encoding is not None:
            known_face_encodings.append(new_face_encoding)
            known_face_names.append(name)
            logger.info(f"Лицо '{name}' добавлено в базу")
            await message.answer(f"Лицо '{name}' успешно добавлено!", reply_markup=main_menu_reply_markup)
        else:
            logger.error(f"Не удалось распознать лицо при сохранении для '{name}'")
            await message.answer(f"Ошибка: не удалось распознать лицо для '{name}'.", reply_markup=main_menu_reply_markup)
            os.remove(file_path)
    except Exception as e:
        logger.error(f"Ошибка при сохранении лица: {str(e)}")
        await message.answer(f"Ошибка при сохранении лица: {str(e)}", reply_markup=main_menu_reply_markup)
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
    
    await state.clear()

# Обработка выбора потока для получения кадра
@dp.message(StateFilter(StreamSelection.waiting_for_stream))
async def select_stream(message: types.Message, state: FSMContext):
    user_id = message.from_user.id
    main_menu_reply_markup = create_main_menu(is_admin(user_id))
    
    if not is_admin(user_id):
        await message.answer("У вас нет прав на получение кадров.", reply_markup=main_menu_reply_markup)
        await state.clear()
        return

    if message.text == "Отмена":
        await message.answer("Выбор отменён.", reply_markup=main_menu_reply_markup)
        await state.clear()
    elif message.text in config["streams"]:
        stream_name = message.text
        frame_requests[stream_name] = message.chat.id  # Записываем запрос
        await message.answer(f"Ожидаю следующий кадр из потока {stream_name}...", reply_markup=main_menu_reply_markup)
        await state.clear()
    else:
        await message.answer("Неверный выбор. Пожалуйста, выберите поток из списка.", reply_markup=create_streams_keyboard())

# Функция для обработки одного видеопотока
async def process_stream(stream_name, stream_url, task_manager):
    global last_frames_paths, cuda_warning_shown, detected_faces, last_seen_time, frame_requests
    cap = None
    try:
        logger.info(f"Подключаюсь к потоку: {stream_name}")
        cap = cv2.VideoCapture(stream_url)
        
        if not cap.isOpened():
            logger.error(f"Не удалось открыть поток: {stream_name}")
            await bot.send_message(chat_id, f"Не удалось открыть поток: {stream_name}")
            await asyncio.sleep(5)
            return
        
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error(f"Не удалось получить кадр из потока: {stream_name}")
                break
            
            # Использование GPU для обработки кадра, если доступно
            if cuda_available:
                try:
                    gpu_frame = cv2.cuda_GpuMat()
                    gpu_frame.upload(frame)
                    gpu_resized = cv2.cuda.resize(gpu_frame, (640, 480))
                    frame_processed = gpu_resized.download()
                except Exception as e:
                    if not cuda_warning_shown:
                        logger.warning(f"Ошибка GPU: {e}. Используется CPU.")
                        cuda_warning_shown = True
                    frame_processed = cv2.resize(frame, (640, 480))
            else:
                frame_processed = cv2.resize(frame, (640, 480))
            
            # Обработка запроса на кадр
            if stream_name in frame_requests:
                frame_path = f"requested_frame_{stream_name}.jpg"
                cv2.imwrite(frame_path, frame_processed)
                photo = FSInputFile(frame_path)
                chat_id_to_send = frame_requests[stream_name]
                await bot.send_photo(chat_id=chat_id_to_send, photo=photo)
                await bot.send_message(chat_id=chat_id_to_send, text="Кадр отправлен.", reply_markup=create_main_menu(is_admin(chat_id_to_send)))
                del frame_requests[stream_name]  # Удаляем запрос после отправки
                logger.info(f"Отправлен запрошенный кадр из потока {stream_name}")
            
            # Распознавание лиц
            try:
                face_locations, face_names = recognize_faces(frame_processed, known_face_encodings, known_face_names)
                current_time = time.time()
                
                # Обновляем время последнего обнаружения для текущих лиц
                current_faces = set(face_names)
                for name in current_faces:
                    last_seen_time[stream_name][name] = current_time
                
                # Проверяем, какие лица исчезли из кадра
                for name in list(last_seen_time[stream_name].keys()):
                    if name not in current_faces and (current_time - last_seen_time[stream_name][name]) >= RESET_TIMEOUT:
                        detected_faces[stream_name].discard(name)
                        del last_seen_time[stream_name][name]
                        logger.info(f"Лицо {name} исчезло из потока {stream_name} более чем на {RESET_TIMEOUT} сек, состояние сброшено")
                
                if face_locations:
                    logger.info(f"Обнаружено {len(face_locations)} лиц в потоке {stream_name}")
                    last_frame_path = f"detected_frame_{stream_name}.jpg"
                    cv2.imwrite(last_frame_path, frame_processed)
                    last_frames_paths[stream_name] = last_frame_path
                    
                    for name in face_names:
                        if name not in detected_faces[stream_name]:  # Первое появление
                            detected_faces[stream_name].add(name)
                            if name != "Unknown":
                                logger.info(f"Обнаружено лицо: {name} в потоке: {stream_name}")
                                message = f"Обнаружено лицо: {name} в потоке: {stream_name}"
                                await bot.send_message(chat_id=chat_id, text=message)
                                photo = FSInputFile(last_frame_path)
                                await bot.send_photo(chat_id=chat_id, photo=photo)
                            else:
                                logger.info(f"Неизвестное лицо в потоке {stream_name}")
                                await bot.send_message(chat_id, f"Обнаружено неизвестное лицо в потоке: {stream_name}")
                                photo = FSInputFile(last_frame_path)
                                await bot.send_photo(chat_id=chat_id, photo=photo)
                else:
                    logger.info(f"Лица не обнаружены в потоке {stream_name}")
            except Exception as e:
                logger.error(f"Ошибка обработки кадра в потоке {stream_name}: {str(e)}")
            
            await asyncio.sleep(0.1)
    
    except Exception as e:
        logger.error(f"Критическая ошибка в потоке {stream_name}: {str(e)}")
    finally:
        if cap is not None:
            cap.release()
        logger.info(f"Поток {stream_name} завершён")
        # Удаляем задачу из task_manager
        if stream_name in task_manager:
            del task_manager[stream_name]
        await asyncio.sleep(5)

# Запуск бота и автоматический запуск обработки потоков
async def main():
    # Словарь для хранения задач
    task_manager = {}
    
    # Запускаем задачи для каждого потока
    for stream_name, stream_url in config["streams"].items():
        task = asyncio.create_task(process_stream(stream_name, stream_url, task_manager))
        task_manager[stream_name] = task
    
    # Запускаем бота
    await dp.start_polling(bot)

    # Ожидаем завершения всех задач
    for task in task_manager.values():
        await task

if __name__ == "__main__":
    asyncio.run(main())