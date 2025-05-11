import cv2
import glob

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

    # No face detected
    return None


def preprocess_image_array(img_array: np.ndarray, target_size=(224, 224)) -> np.ndarray:
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


def predict_detected_faces(model, class_names: list[str], padding: float, image_directory: str = '../Images/*.jpg') -> None:
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
        face_cropped = detect_face(image_path, padding)

        if face_cropped is not None:
            # Convert the face to RGB for displaying
            face_rgb = cv2.cvtColor(face_cropped, cv2.COLOR_BGR2RGB)

            # Preprocess the face image for prediction
            preprocessed = preprocess_image_array(face_rgb, target_size=(224, 224))

            # Predict the feature
            prediction = model.predict(preprocessed)

            # Convert prediction to label
            predicted_labels = (prediction > 0.5).astype(int).flatten()
            predicted_label = class_names[predicted_labels[0]]

            # Display the cropped face and predicted label
            plt.imshow(face_rgb)
            plt.title(f"Prediction: {predicted_label} ({prediction[0][0]:.2f})")
            plt.axis("off")
            plt.show()
        else:
            print(f"No face detected in image: {image_path}")