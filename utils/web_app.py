import logging
import os
from threading import Lock

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from utils.yolo_utils import save_known_face

logger = logging.getLogger(__name__)


def create_web_app(
    known_face_encodings,
    known_face_names,
    known_faces_lock: Lock,
    known_faces_path: str,
    validate_web_code,
):
    app = FastAPI(title="Face Recognition Bot", version="1.0")

    @app.get("/health")
    async def healthcheck():
        return {"status": "ok"}

    @app.post("/users")
    async def add_user(
        name: str = Form(...),
        photo: UploadFile = File(...),
        username: str = Form(...),
        code: str = Form(...),
    ):
        if not validate_web_code(username, code):
            raise HTTPException(status_code=403, detail="Недействительный код или нет прав")

        if photo.content_type not in {"image/jpeg", "image/png"}:
            raise HTTPException(status_code=400, detail="Поддерживаются только JPEG/PNG")

        os.makedirs(known_faces_path, exist_ok=True)
        file_path = os.path.join(known_faces_path, f"{name}.jpg")
        try:
            content = await photo.read()
            with open(file_path, "wb") as buffer:
                buffer.write(content)
            logger.info("Получено новое фото для пользователя %s", name)

            new_encoding, _ = save_known_face(file_path)
            if new_encoding is None:
                os.remove(file_path)
                raise HTTPException(status_code=400, detail="Лицо не обнаружено на фото")

            with known_faces_lock:
                if name in known_face_names:
                    existing_index = known_face_names.index(name)
                    known_face_encodings[existing_index] = new_encoding
                else:
                    known_face_encodings.append(new_encoding)
                    known_face_names.append(name)
        except HTTPException:
            raise
        except Exception as exc:  # pragma: no cover - сетевые/IO ошибки
            logger.error("Ошибка при сохранении пользователя %s: %s", name, exc)
            if os.path.exists(file_path):
                os.remove(file_path)
            raise HTTPException(status_code=500, detail="Не удалось сохранить пользователя")

        return JSONResponse({"status": "ok", "user": name})

    return app
