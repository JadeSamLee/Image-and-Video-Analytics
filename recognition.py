import cv2
import numpy as np
import os
from typing import Dict, List, Tuple

def capture_images(save_dir: str) -> Tuple[str, str]:
    """
    Capture 5 images from the webcam and save them under the user's name directory.

    Args:
        save_dir (str): Base directory to store training images.

    Returns:
        Tuple[str, str]: User's name and directory path where images are saved.

    Raises:
        ValueError: If webcam fails to open or directory creation fails.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise ValueError("Failed to open webcam.")

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        cap.release()
        raise ValueError("Failed to load Haar Cascade classifier.")

    print("Enter your name: ", end="")
    user_name = input().strip()
    user_dir = os.path.join(save_dir, user_name)
    os.makedirs(user_dir, exist_ok=True)

    print("Capturing 5 images. Look at the camera and press 'c' to capture each image.")
    captured = 0
    while captured < 5:
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Capture Face", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c') and len(faces) > 0:
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (100, 100))  # Standardize size
            image_path = os.path.join(user_dir, f"{user_name}_{captured + 1}.jpg")
            cv2.imwrite(image_path, face_roi)
            print(f"Captured image {captured + 1}/5: {image_path}")
            captured += 1
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return user_name, user_dir

def load_training_data(training_dir: str) -> Tuple[List[np.ndarray], List[int], Dict[int, str]]:
    """
    Load face images and labels from the training directory.

    Args:
        training_dir (str): Directory containing subdirectories of face images.

    Returns:
        Tuple[List[np.ndarray], List[int], Dict[int, str]]: Lists of face images, labels, and label-to-name mapping.

    Raises:
        ValueError: If no valid images are found.
    """
    faces = []
    labels = []
    label_dict = {}
    current_id = 0

    for person_name in os.listdir(training_dir):
        person_dir = os.path.join(training_dir, person_name)
        if not os.path.isdir(person_dir):
            continue
        label_dict[current_id] = person_name
        for image_name in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None or image.size == 0:
                continue
            faces.append(image)
            labels.append(current_id)
        current_id += 1

    if not faces:
        raise ValueError(f"No faces found in training directory {training_dir}")
    return faces, labels, label_dict

def train_recognizer(faces: List[np.ndarray], labels: List[int]) -> cv2.face.LBPHFaceRecognizer:
    """
    Train an LBPH face recognizer with the provided face images and labels.

    Args:
        faces (List[np.ndarray]): List of face images.
        labels (List[int]): List of corresponding labels.

    Returns:
        cv2.face.LBPHFaceRecognizer: Trained face recognizer.
    """
    recognizer = cv2.face.LBPHFaceRecognizer_create()  # Corrected method
    recognizer.train(faces, np.array(labels))
    return recognizer

def recognize_faces_in_video(recognizer: cv2.face.LBPHFaceRecognizer, label_dict: Dict[int, str]) -> None:
    """
    Recognize faces in a live video feed and display the recognized names.

    Args:
        recognizer (cv2.face.LBPHFaceRecognizer): Trained face recognizer.
        label_dict (Dict[int, str]): Mapping of label IDs to names.

    Raises:
        ValueError: If webcam fails to open.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise ValueError("Failed to open webcam.")

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        cap.release()
        raise ValueError("Failed to load Haar Cascade classifier.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (100, 100))  # Match training size
            label, confidence = recognizer.predict(roi_gray)
            name = label_dict.get(label, "Unknown") if confidence < 100 else "Unknown"
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            print(f"Recognized face: {name} (Confidence: {confidence:.2f})")

        cv2.imshow("Face Recognition in Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main() -> None:
    """Main function to capture, train, and recognize faces in a video feed."""
    training_dir = "training_faces"  # Directory for all training images
    os.makedirs(training_dir, exist_ok=True)

    try:
        # Capture and save user's images
        user_name, user_dir = capture_images(training_dir)

        # Load and train with all face data
        faces, labels, label_dict = load_training_data(training_dir)
        recognizer = train_recognizer(faces, labels)

        # Recognize faces in live video
        recognize_faces_in_video(recognizer, label_dict)

    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()