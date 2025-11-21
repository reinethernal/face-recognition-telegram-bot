import asyncio
import os
import cv2
import yaml
import logging
import importlib.util
import secrets
import string
import time
from threading import Lock, Thread
from typing import Dict, Iterable, List, Set

from aiogram import Bot, Dispatcher, types
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton, ContentType, FSInputFile
from aiogram.enums import ParseMode
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.filters import Command, StateFilter
from aiogram.client.default import DefaultBotProperties
import uvicorn

from utils.yolo_utils import recognize_faces, load_known_faces, save_known_face
from utils.web_app import create_web_app

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Загрузка конфигурации из YAML файла
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Инициализация бота и диспетчера
bot_token = config["telegram"]["bot_token"]
chat_id = config["telegram"]["chat_id"]
admin_ids = config["telegram"].get("admin_ids", [])
admin_usernames_path = config["telegram"].get("admin_usernames_path", "admins.yaml")
initial_admin_usernames = set(config["telegram"].get("admin_usernames", []))
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

yolo_config = config.get("yolo", {})
yolo_model_path = yolo_config.get("model_path", "yolo11n-face.pt")
yolo_confidence = float(yolo_config.get("confidence", 0.35))
stable_frame_count = int(yolo_config.get("stable_frames", 3))

web_config = config.get("web", {})
web_host = web_config.get("host", "0.0.0.0")
web_port = int(web_config.get("port", 8000))

known_faces_lock = Lock()
admin_lock = Lock()
web_code_lock = Lock()

# Загрузка известных лиц
known_face_encodings, known_face_names = load_known_faces(known_faces_path)

def _load_admin_usernames(path: str, defaults: Iterable[str]) -> Set[str]:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            usernames: List[str] = data.get("usernames", [])
            return {normalize_username(u) for u in usernames}
    return {normalize_username(u) for u in defaults}


def _save_admin_usernames(path: str, usernames: Iterable[str]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump({"usernames": sorted({normalize_username(u) for u in usernames})}, f, allow_unicode=True)


def normalize_username(username: str) -> str:
    return username.lstrip("@").lower()


admin_usernames: Set[str] = _load_admin_usernames(admin_usernames_path, initial_admin_usernames)
if not os.path.exists(admin_usernames_path) and admin_usernames:
    _save_admin_usernames(admin_usernames_path, admin_usernames)

# Словарь для хранения пути к последнему кадру для каждого потока
last_frames_paths = {stream: None for stream in config["streams"].keys()}

# Словарь для отслеживания обнаруженных лиц (поток: множество имён)
detected_faces = {stream: set() for stream in config["streams"].keys()}

# Словарь для отслеживания времени последнего обнаружения лица (поток: {имя: время})
last_seen_time = {stream: {} for stream in config["streams"].keys()}

# Счётчики стабильных детекций (поток: {имя: счётчик})
detection_streaks: Dict[str, Dict[str, int]] = {stream: {} for stream in config["streams"].keys()}

# Словарь для запросов на получение кадров (поток: chat_id)
frame_requests = {}

# Временные коды авторизации для веб-интерфейса (username -> {code, expires})
web_login_codes: Dict[str, Dict[str, float]] = {}

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


def start_web_interface():
    app = create_web_app(
        known_face_encodings,
        known_face_names,
        known_faces_lock,
        known_faces_path,
        validate_web_code,
    )
    config_uvicorn = uvicorn.Config(app=app, host=web_host, port=web_port, log_level="info")
    server = uvicorn.Server(config_uvicorn)
    thread = Thread(target=server.run, daemon=True)
    thread.start()
    logger.info(f"Веб-интерфейс запущен: http://{web_host}:{web_port}")
    return thread

# Состояния для FSM
class AddFace(StatesGroup):
    waiting_for_photo = State()
    waiting_for_name = State()

class StreamSelection(StatesGroup):
    waiting_for_stream = State()


class AdminManagement(StatesGroup):
    waiting_for_admins = State()

# Функция для создания главного меню с учётом прав
def create_main_menu(is_admin=False):
    keyboard = [
        [KeyboardButton(text="Начать")],
    ]
    if is_admin:
        keyboard.extend([
            [KeyboardButton(text="Добавить лицо")],
            [KeyboardButton(text="Получить кадр")],
            [KeyboardButton(text="Администраторы")],
            [KeyboardButton(text="Код для веб")],
        ])
        for plugin_name in plugins.keys():
            keyboard.append([KeyboardButton(text=f"Плагин: {plugin_name}")])
    return ReplyKeyboardMarkup(keyboard=keyboard, resize_keyboard=True)

# Функция для создания клавиатуры выбора потоков
def create_streams_keyboard():
    keyboard = [[KeyboardButton(text=str(stream))] for stream in config["streams"].keys()]
    keyboard.append([KeyboardButton(text="Отмена")])
    return ReplyKeyboardMarkup(keyboard=keyboard, resize_keyboard=True)


def format_admins_list() -> str:
    if not admin_usernames:
        return "Список администраторов пуст."
    sorted_admins = sorted(admin_usernames)
    return "\n".join(f"- @{name}" for name in sorted_admins)


async def prompt_admins_update(message: types.Message, state: FSMContext):
    if not is_admin_user(message.from_user):
        await message.answer("У вас нет прав на управление администраторами.", reply_markup=create_main_menu())
        await state.clear()
        return

    current_admins = format_admins_list()
    await message.answer(
        "Текущий список администраторов:\n"
        f"{current_admins}\n\n"
        "Отправьте usernames (через пробел или с новой строки) для установки нового списка."
        " Ваш username будет автоматически добавлен.",
        reply_markup=create_main_menu(True),
    )
    await state.set_state(AdminManagement.waiting_for_admins)


async def send_web_code(message: types.Message):
    try:
        if not is_admin_user(message.from_user):
            await message.answer("У вас нет прав на получение кода.", reply_markup=create_main_menu())
            return
        username = require_username(message.from_user)
        code = generate_web_code(username)
        await message.answer(
            f"Код для входа в веб-интерфейс: <b>{code}</b>\n"
            "Время действия — 5 минут. Введите его вместе со своим username в форме входа веб-интерфейса.",
            reply_markup=create_main_menu(True),
        )
    except ValueError as err:
        await message.answer(str(err), reply_markup=create_main_menu())

# Проверка, является ли пользователь администратором
def is_admin_user(user: types.User) -> bool:
    if user.id in admin_ids:
        return True
    if user.username:
        return normalize_username(user.username) in admin_usernames
    return False


def require_username(user: types.User) -> str:
    if not user.username:
        raise ValueError("У вас не указан username в Telegram. Установите его в настройках и попробуйте снова.")
    return normalize_username(user.username)


def add_admin_usernames(usernames: Iterable[str]) -> Set[str]:
    normalized = {normalize_username(u) for u in usernames if u}
    with admin_lock:
        admin_usernames.update(normalized)
        _save_admin_usernames(admin_usernames_path, admin_usernames)
        return set(admin_usernames)


def replace_admin_usernames(usernames: Iterable[str]) -> Set[str]:
    normalized = {normalize_username(u) for u in usernames if u}
    with admin_lock:
        admin_usernames.clear()
        admin_usernames.update(normalized)
        _save_admin_usernames(admin_usernames_path, admin_usernames)
        return set(admin_usernames)


def generate_web_code(username: str, ttl_seconds: int = 300) -> str:
    code = "".join(secrets.choice(string.digits) for _ in range(6))
    expires = time.time() + ttl_seconds
    with web_code_lock:
        web_login_codes[normalize_username(username)] = {"code": code, "expires": expires}
    return code


def validate_web_code(username: str, code: str) -> bool:
    normalized = normalize_username(username)
    if normalized not in admin_usernames:
        return False
    with web_code_lock:
        entry = web_login_codes.get(normalized)
        if not entry:
            return False
        if entry["code"] != code:
            return False
        if time.time() > entry["expires"]:
            del web_login_codes[normalized]
            return False
        del web_login_codes[normalized]
        return True

# Обработка команды /start
@dp.message(Command(commands=["start"]))
async def start(message: types.Message, state: FSMContext):
    main_menu_reply_markup = create_main_menu(is_admin_user(message.from_user))
    await state.clear()  # Очищаем состояние при старте
    await message.answer("Добро пожаловать! Обработка потоков уже запущена. Выберите действие:", reply_markup=main_menu_reply_markup)


@dp.message(Command(commands=["admins"]))
async def admins_command(message: types.Message, state: FSMContext):
    await prompt_admins_update(message, state)


@dp.message(Command(commands=["webcode"]))
async def webcode_command(message: types.Message):
    await send_web_code(message)

# Обработка выбора команды через кнопки
@dp.message(
    lambda message: message.text
    and (
        message.text in ["Начать", "Добавить лицо", "Получить кадр", "Администраторы", "Код для веб"]
        or message.text.startswith("Плагин: ")
    )
)
async def handle_choice(message: types.Message, state: FSMContext):
    main_menu_reply_markup = create_main_menu(is_admin_user(message.from_user))

    if not is_admin_user(message.from_user) and message.text != "Начать":
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
    elif choice == "Администраторы":
        await prompt_admins_update(message, state)
    elif choice == "Код для веб":
        await send_web_code(message)
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
    main_menu_reply_markup = create_main_menu(is_admin_user(message.from_user))

    if not is_admin_user(message.from_user):
        await message.answer("У вас нет прав на добавление лиц.", reply_markup=main_menu_reply_markup)
        await state.clear()
        return

    temp_file_path = os.path.join(known_faces_path, f"temp_{message.from_user.id}_{message.message_id}.jpg")

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
    main_menu_reply_markup = create_main_menu(is_admin_user(message.from_user))

    if not is_admin_user(message.from_user):
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
            with known_faces_lock:
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


@dp.message(StateFilter(AdminManagement.waiting_for_admins), lambda message: message.text is not None)
async def update_admins(message: types.Message, state: FSMContext):
    main_menu_reply_markup = create_main_menu(is_admin_user(message.from_user))

    if not is_admin_user(message.from_user):
        await message.answer("У вас нет прав на изменение списка администраторов.", reply_markup=main_menu_reply_markup)
        await state.clear()
        return

    usernames_raw = {part.strip() for part in message.text.replace("\n", " ").split(" ") if part.strip()}
    try:
        sender_username = require_username(message.from_user)
    except ValueError as err:
        await message.answer(str(err), reply_markup=main_menu_reply_markup)
        await state.clear()
        return

    updated = replace_admin_usernames(usernames_raw | {sender_username})
    await message.answer(
        "Список администраторов обновлён:\n" + "\n".join(f"- @{u}" for u in sorted(updated)),
        reply_markup=main_menu_reply_markup,
    )
    await state.clear()

# Обработка выбора потока для получения кадра
@dp.message(StateFilter(StreamSelection.waiting_for_stream))
async def select_stream(message: types.Message, state: FSMContext):
    main_menu_reply_markup = create_main_menu(is_admin_user(message.from_user))

    if not is_admin_user(message.from_user):
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
    global last_frames_paths, cuda_warning_shown, detected_faces, last_seen_time, frame_requests, detection_streaks
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
                await bot.send_message(chat_id=chat_id_to_send, text="Кадр отправлен.", reply_markup=create_main_menu())
                del frame_requests[stream_name]  # Удаляем запрос после отправки
                logger.info(f"Отправлен запрошенный кадр из потока {stream_name}")
            
            # Распознавание лиц
            try:
                with known_faces_lock:
                    encodings_snapshot = list(known_face_encodings)
                    names_snapshot = list(known_face_names)

                face_locations, face_names = recognize_faces(
                    frame_processed,
                    encodings_snapshot,
                    names_snapshot,
                    yolo_model_path,
                    yolo_confidence,
                )
                current_time = time.time()

                current_faces = set(face_names)
                for name in current_faces:
                    last_seen_time[stream_name][name] = current_time
                    detection_streaks[stream_name][name] = detection_streaks[stream_name].get(name, 0) + 1

                for name in list(last_seen_time[stream_name].keys()):
                    if name not in current_faces and (current_time - last_seen_time[stream_name][name]) >= RESET_TIMEOUT:
                        detected_faces[stream_name].discard(name)
                        detection_streaks[stream_name].pop(name, None)
                        del last_seen_time[stream_name][name]
                        logger.info(
                            f"Лицо {name} исчезло из потока {stream_name} более чем на {RESET_TIMEOUT} сек, состояние сброшено"
                        )

                for name in list(detection_streaks[stream_name].keys()):
                    if name not in current_faces:
                        detection_streaks[stream_name][name] = 0

                if face_locations:
                    logger.info(f"Обнаружено {len(face_locations)} лиц в потоке {stream_name}")
                    last_frame_path = f"detected_frame_{stream_name}.jpg"
                    cv2.imwrite(last_frame_path, frame_processed)
                    last_frames_paths[stream_name] = last_frame_path

                    for name in face_names:
                        if detection_streaks[stream_name].get(name, 0) < stable_frame_count:
                            continue

                        if name not in detected_faces[stream_name]:  # Первое появление после подтверждения
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

    # Запуск веб-интерфейса для добавления пользователей
    start_web_interface()

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