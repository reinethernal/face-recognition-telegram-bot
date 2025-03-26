Вот пример описания для вашего бота на русском языке, которое можно разместить на GitHub. Оно включает краткое описание, функционал, инструкции по установке и использованию, а также структуру проекта. Вы можете адаптировать его под свои нужды.

---

# Face Recognition Telegram Bot

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Описание

**Face Recognition Telegram Bot** — это Telegram-бот, который выполняет распознавание лиц в видеопотоках (RTSP) с использованием компьютерного зрения. Бот отправляет уведомления в Telegram при обнаружении известных или неизвестных лиц, позволяет администраторам добавлять новые лица в базу, запрашивать кадры из потоков и использовать плагины для расширения функционала.

Проект использует библиотеки `aiogram` для работы с Telegram API, `OpenCV` для обработки видеопотоков и кастомный модуль `yolo_utils` для распознавания лиц (можно заменить на `face_recognition` или другой инструмент).

## Основные возможности

- **Распознавание лиц**: Автоматическое обнаружение лиц в видеопотоках с уведомлениями в Telegram.
- **Управление через Telegram**:
  - Добавление новых лиц в базу (для администраторов).
  - Запрос текущего кадра из выбранного потока.
  - Использование плагинов (например, подсчёт уникальных лиц).
- **Система прав**: Доступ к функциям ограничен для администраторов (настраивается через `config.yaml`).
- **Фоновая работа**: Бот работает как сервис (`systemd`), обрабатывая несколько видеопотоков одновременно.
- **Логирование**: Подробные логи для отладки и мониторинга.

## Требования

- Python 3.10 или выше
- Linux-сервер (Ubuntu/Debian рекомендуется)
- Доступ к RTSP-потокам (видеокамеры)
- Telegram-бот (токен от @BotFather)
- (Опционально) GPU с поддержкой CUDA для ускорения обработки

## Установка

1. **Клонируйте репозиторий**:
   ```bash
   git clone https://github.com/yourusername/face-recognition-telegram-bot.git
   cd face-recognition-telegram-bot
   ```

2. **Установите зависимости**:
   ```bash
   sudo apt update
   sudo apt install python3 python3-pip python3-venv libopencv-dev python3-opencv -y
   python3 -m venv venv
   source venv/bin/activate
   pip install aiogram opencv-python pyyaml
   ```

3. **Настройте `config.yaml`**:
   - Скопируйте пример конфигурации:
     ```bash
     cp config.example.yaml config.yaml
     ```
   - Отредактируйте `config.yaml`:
     ```yaml
     telegram:
       bot_token: "your_bot_token_here"
       chat_id: "your_chat_id_here"
       admin_ids:
         - 123456789
         - 987654321
     streams:
       stream1: "rtsp://your_stream_url_1"
       stream2: "rtsp://your_stream_url_2"
     known_faces_path: "known_faces"
     ```
     - `bot_token`: Токен вашего бота от @BotFather.
     - `chat_id`: ID чата, куда будут отправляться уведомления.
     - `admin_ids`: Список Telegram ID администраторов.
     - `streams`: Список RTSP-URL ваших видеопотоков.
     - `known_faces_path`: Папка для хранения изображений известных лиц.

4. **Создайте папку для известных лиц**:
   ```bash
   mkdir known_faces
   ```
   Добавьте изображения лиц в формате `имя_лица.jpg` (например, `John.jpg`).

5. **Запустите бота**:
   ```bash
   python3 main.py
   ```

6. **(Опционально) Настройте как сервис**:
   - Создайте файл сервиса:
     ```bash
     sudo nano /etc/systemd/system/face_recognition_bot.service
     ```
     Вставьте:
     ```ini
     [Unit]
     Description=Face Recognition Telegram Bot
     After=network.target

     [Service]
     User=your_username
     Group=your_username
     WorkingDirectory=/path/to/face-recognition-telegram-bot
     ExecStart=/path/to/face-recognition-telegram-bot/venv/bin/python3 /path/to/face-recognition-telegram-bot/main.py
     Restart=always
     RestartSec=10
     KillMode=process

     [Install]
     WantedBy=multi-user.target
     ```
   - Активируйте сервис:
     ```bash
     sudo systemctl daemon-reload
     sudo systemctl start face_recognition_bot
     sudo systemctl enable face_recognition_bot
     ```

## Использование

1. **Запустите бота**:
   - Если настроен как сервис, он уже работает.
   - Или запустите вручную: `python3 main.py`.

2. **Команды в Telegram**:
   - `/start`: Запускает бота и показывает меню.
   - **Для администраторов**:
     - `Добавить лицо`: Добавляет новое лицо в базу (отправьте фото и укажите имя).
     - `Получить кадр`: Запрашивает текущий кадр из выбранного потока.
     - `Плагин: face_counter`: Показывает количество уникальных обнаруженных лиц.

3. **Уведомления**:
   - При обнаружении лица бот отправляет сообщение в указанный `chat_id` с фото и именем (если лицо известно).

## Структура проекта

```
face-recognition-telegram-bot/
│
├── main.py              # Основной файл бота
├── config.yaml          # Конфигурация (токен, потоки, admin_ids)
├── config.example.yaml  # Пример конфигурации
├── known_faces/         # Папка для хранения изображений известных лиц
├── plugins/             # Папка для плагинов
│   └── face_counter.py  # Пример плагина (подсчёт лиц)
├── utils/               # Утилиты
│   └── yolo_utils.py    # Модуль для распознавания лиц (кастомный)
└── venv/                # Виртуальное окружение
```

## Плагины

Бот поддерживает плагины для расширения функционала. Пример плагина `face_counter.py` подсчитывает количество уникальных лиц. Чтобы добавить новый плагин:

1. Создайте файл в папке `plugins/`, например, `my_plugin.py`.
2. Реализуйте функции `setup()` и `execute()`:
   ```python
   import logging

   logger = logging.getLogger(__name__)

   def setup():
       logger.info("Плагин my_plugin загружен")

   async def execute(bot, message, chat_id, detected_faces):
       return "Мой плагин работает!"
   ```
3. Плагин автоматически появится в меню администраторов.

## Лицензия

Проект распространяется под лицензией MIT. См. файл [LICENSE](LICENSE) для подробностей.

---

### Примечания
- Замените `yourusername` в ссылке на репозиторий на ваш GitHub-ник.
- Добавьте файл `LICENSE` (например, MIT) в репозиторий.
- Если у вас есть дополнительные зависимости или особенности (например, использование YOLO вместо `face_recognition`), уточните это в описании.
- Если хотите, можно добавить скриншоты работы бота в Telegram в раздел "Использование".

Готово! Если нужно что-то изменить или добавить, дайте знать!
