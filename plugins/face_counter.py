import logging
from aiogram import Bot, types

logger = logging.getLogger(__name__)

def setup():
    logger.info("Плагин face_counter загружен")

async def execute(bot: Bot, message: types.Message, chat_id: str, detected_faces) -> str:
    total_faces = set()
    for stream_name, faces in detected_faces.items():
        total_faces.update(faces)
    count = len(total_faces)
    return f"Общее количество уникальных обнаруженных лиц: {count}"