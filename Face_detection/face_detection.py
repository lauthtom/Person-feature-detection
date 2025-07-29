import cv2
import glob
import mediapipe as mp

import numpy as np
import matplotlib.pyplot as plt


def detect_face(image_path: str, padding: float) -> np.ndarray | None:
    """
    Detects the first face in the image and returns the cropped face region with optional padding.

    Parameters
    ----------
    image_path : str
        Path to the input image.

    padding : float
        Relative amount of padding to apply around the detected face bounding box.
        For example, 0.2 will expand the bounding box by 20% on each side.

    Returns
    -------
    np.ndarray or None
        Cropped image of the detected face as a NumPy array (BGR format).
        Returns None if no face is detected.
    """
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Load the face classifier
    face_classifier = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    # Detect face
    faces = face_classifier.detectMultiScale(
        gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    )

    # padding = 0.2
    """
    for (x, y, w, h) in faces:
        # Expand bounding box
        x_pad = int(padding * w)
        y_pad = int(padding * h)

        x_new = max(x - x_pad, 0)
        y_new = max(y - y_pad, 0)
        w_new = min(w + 2 * x_pad, img.shape[1] - x_new)
        h_new = min(h + 2 * y_pad, img.shape[0] - y_new)

        face_cropped = img[y_new:y_new + h_new, x_new:x_new + w_new]
        return face_cropped
    """
    for (x, y, w, h) in faces:
        x_pad = int(padding * w)
        y_pad = int(padding * h)

        # Haarregion nach oben ausdehnen
        hair_extension = int(h * 0.4)

        x_new = max(x - x_pad, 0)
        y_new = max(y - y_pad - hair_extension, 0)
        w_new = min(w + 2 * x_pad, img.shape[1] - x_new)
        h_new = min(h + 2 * y_pad + hair_extension, img.shape[0] - y_new)

        face_with_hair = img[y_new:y_new + h_new, x_new:x_new + w_new]
        return face_with_hair

    # No face detected
    return None


def detect_face_mediapipe(image_path: str, hair_extension_ratio=0.7):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    mp_face_detection = mp.solutions.face_detection
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.6) as face_detection:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_detection.process(img_rgb)

        if not results.detections:
            return None

        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box
        h, w, _ = img.shape

        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        w_box = int(bbox.width * w)
        h_box = int(bbox.height * h)

        y = max(0, y - int(h_box * hair_extension_ratio))
        h_box = min(h - y, h_box + int(h_box * hair_extension_ratio))

        return img[y:y + h_box, x:x + w_box]


def detect_face_from_frame(frame: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]] | tuple[None, None]:
    """
    Detects a face in a given image frame and returns the cropped face region along with its bounding box coordinates.

    Parameters
    ----------
    frame : np.ndarray
        The input image frame in which to detect a face. Expected to be a color image (BGR).

    Returns
    -------
    tuple[np.ndarray, tuple[int, int, int, int]] or tuple[None, None]
        If a face is detected, returns a tuple containing:
            - face_cropped : np.ndarray
                The cropped image of the detected face region.
            - (x, y, w, h) : tuple of int
                The coordinates (x, y) of the top-left corner and the width (w) and height (h) of the bounding box around the face.
        If no face is detected, returns (None, None)
    """

    # Tried other classifier from google
    mp_face_detection = mp.solutions.face_detection
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.6) as face_detection:
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(img_rgb)

        if not results.detections:
            return None, None

        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box
        h, w, _ = frame.shape

        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        w_box = int(bbox.width * w)
        h_box = int(bbox.height * h)

        hair_extension_ratio = 0.3
        y = max(0, y - int(h_box * hair_extension_ratio))
        h_box = min(h - y, h_box + int(h_box * hair_extension_ratio))
        x = max(0, x)
        w_box = min(w - x, w_box)

        face_cropped = frame[y:y + h_box, x:x + w_box]
        return face_cropped, (x, y, w_box, h_box)

    return None, None


    # gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # face_classifier = cv2.CascadeClassifier(
    #     cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    # )

    # faces = face_classifier.detectMultiScale(
    #     gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    # )

    # for (x, y, w, h) in faces:
    #     x_pad = int(0.1 * w)
    #     y_pad = int(0.3 * h)

    #     x_new = max(x - x_pad, 0)
    #     y_new = max(y - y_pad, 0)
    #     w_new = min(w + 2 * x_pad, frame.shape[1] - x_new)
    #     h_new = min(h + 2 * y_pad, frame.shape[0] - y_new)

    #     face_cropped = frame[y_new:y_new + h_new, x_new:x_new + w_new]
    #     return face_cropped, (x_new, y_new, w_new, h_new)

    # return None, None



def preprocess_image_array(img_array: np.ndarray, target_size: tuple = (224, 224)) -> np.ndarray:
    """
    Resizes, normalizes and expands dimensions of an image array.

    Parameters
    ----------
    img_array : np.ndarray
        Image array (BGR or RGB).

    target_size : tuple
        Desired size of the input image.

    Returns
    -------
    np.ndarray
        Preprocessed image ready for model prediction.
    """
    img = cv2.resize(img_array, target_size)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    return img


def predict_detected_faces(model, class_names: list[str], padding: float, categorical: bool, image_directory: str = '../Images/*.jpg') -> None:
    """
    Predicts the gender of faces detected in images from a specified directory.

    This function loops through all images in the specified directory, detects faces in each
    image, preprocesses the detected face, and passes it through the model to predict the gender.

    Parameters
    ----------
    image_directory : str, optional, default='../Images/*.jpg'
        Path pattern to load images

    model : keras.Model
        The trained model to use for predicting the cropped face images.

    padding : float
        Relative amount of padding to apply around the detected face bounding box.
        For example, 0.2 will expand the bounding box by 20% on each side.


    Returns
    -------
    None
        This function displays the predicted gender for each image with the respective cropped face.
    """
    # Get all images in the directory
    images = glob.glob(image_directory)

    for i, image_path in enumerate(images):
        # Detect face and crop the face region
        # face_cropped = detect_face(image_path, padding)
        face_cropped = detect_face_mediapipe(image_path)

        if face_cropped is not None:
            # Convert the face to RGB for displaying
            face_rgb = cv2.cvtColor(face_cropped, cv2.COLOR_BGR2RGB)

            # Preprocess the face image for prediction
            preprocessed = preprocess_image_array(
                face_rgb, target_size=(224, 224))

            # Predict the feature
            prediction = model.predict(preprocessed)
            predicted_label_index = 0

            # Convert prediction to label
            if categorical:
                predicted_label_index = np.argmax(prediction, axis=1)[0]
                predicted_label = class_names[predicted_label_index]
            else:
                predicted_labels = (prediction > 0.5).astype(int).flatten()
                predicted_label = class_names[predicted_labels[0]]

            value = prediction[0][0] if categorical else prediction[0][predicted_label_index]
            # Display the cropped face and predicted label
            plt.imshow(face_rgb)
            plt.title(f"Prediction: {predicted_label} {value}")
            plt.axis("off")
            plt.show()
        else:
            print(f"No face detected in image: {image_path}")
