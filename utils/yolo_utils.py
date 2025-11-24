# utils/yolo_utils.py
import os
import logging
from typing import List, Tuple

import cv2
from ultralytics import YOLO

try:
    import face_recognition
except ImportError as exc:  # pragma: no cover - defensive import guard
    raise ImportError(
        "Не установлен модуль face_recognition. Установите зависимости через "
        "`pip install -r requirements.txt` и убедитесь, что установлены "
        "системные пакеты сборки dlib (например: `sudo apt install build-essential "
        "cmake libopenblas-dev -y`)."
    ) from exc

logger = logging.getLogger(__name__)

_yolo_model = None
_yolo_model_path = None


def _load_yolo_model(model_path: str) -> YOLO:
    global _yolo_model, _yolo_model_path
    if _yolo_model is None or _yolo_model_path != model_path:
        logger.info("Загружаю YOLO11 модель для детекции лиц: %s", model_path)
        _yolo_model = YOLO(model_path)
        _yolo_model_path = model_path
    return _yolo_model


def load_known_faces(known_faces_path: str, log_known_faces: bool = True):
    known_face_encodings = []
    known_face_names = []
    for filename in os.listdir(known_faces_path):
        if filename.endswith(".jpg"):
            name = os.path.splitext(filename)[0]
            image_path = os.path.join(known_faces_path, filename)
            encoding, _ = save_known_face(image_path)
            if encoding is not None:
                known_face_encodings.append(encoding)
                known_face_names.append(name)
                if log_known_faces:
                    logger.info("Загружено лицо: %s из %s", name, image_path)
    return known_face_encodings, known_face_names


def save_known_face(image_path: str):
    image = face_recognition.load_image_file(image_path)
    face_encodings = face_recognition.face_encodings(image)
    if len(face_encodings) > 0:
        logger.info("Лицо найдено в %s, кодировка успешно создана", image_path)
        return face_encodings[0], image
    return None, None


def recognize_faces(
    frame,
    known_face_encodings,
    known_face_names,
    model_path: str,
    confidence: float = 0.35,
) -> Tuple[List[Tuple[int, int, int, int]], List[str]]:
    """
    Детектирует лица с помощью YOLO11 и сопоставляет их с известными кодировками.

    :param frame: кадр в формате BGR
    :param known_face_encodings: список известных кодировок лиц
    :param known_face_names: список имён лиц, соответствующих кодировкам
    :param model_path: путь к весам YOLO11
    :param confidence: минимальный порог уверенности для детекции лица
    :return: список локаций лиц (top, right, bottom, left) и имена
    """

    model = _load_yolo_model(model_path)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = model.predict(source=frame, verbose=False, conf=confidence)
    if not results:
        return [], []

    face_locations: List[Tuple[int, int, int, int]] = []
    for box in results[0].boxes:
        if box.conf is None or float(box.conf[0]) < confidence:
            continue
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        face_locations.append((int(y1), int(x2), int(y2), int(x1)))

    if not face_locations:
        return [], []

    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    face_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
        face_names.append(name)

    return face_locations, face_names
