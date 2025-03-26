# utils/yolo_utils.py
import os
import logging
import cv2  # Добавлен импорт cv2
import face_recognition

logger = logging.getLogger(__name__)

def load_known_faces(known_faces_path, log_known_faces=True):
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
                    logger.info(f"Загружено лицо: {name} из {image_path}")
    return known_face_encodings, known_face_names

def save_known_face(image_path):
    image = face_recognition.load_image_file(image_path)
    face_encodings = face_recognition.face_encodings(image)
    if len(face_encodings) > 0:
        logger.info(f"Лицо найдено в {image_path}, кодировка успешно создана")
        return face_encodings[0], image
    return None, None

def recognize_faces(frame, known_face_encodings, known_face_names):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
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