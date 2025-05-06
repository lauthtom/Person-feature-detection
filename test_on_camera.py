import cv2
import os

import numpy as np
import tensorflow as tf

from tqdm import tqdm

model_names = [
    "Gender_classification.keras",
    "Beard_classification.keras",
    "Haircolor_classification.keras",
    "Nation_classification.keras",
    "Glasses_classification.keras",
]

models_dir = "Models/"
path_to_models = "/Users/tomlauth/Library/CloudStorage/GoogleDrive-santoxhd@gmail.com/Meine Ablage/Studium/Master/2 FS SoSe/Individual Profiling/Old_Models"

target_size = (224, 224)

# All labels for each feature with more than 2 values
haircolor_labels = ["Black hair", "Blond hair", "Brown hair", "Gray hair"]
nation_labels = ["Asian", "Black", "Indian", "Others", "White"]

if len(os.listdir(path_to_models)) == len(model_names):

    models = {}

    for model_name in model_names:
        model_path = os.path.join(path_to_models, model_name)
        print(f"Trying to load the {model_name} model...")
        models[model_name] = tf.keras.models.load_model(model_path)
        print(f"Successfully loaded the {model_name} model.")

    print("All models loaded successfully!")

    answer = input(
        "Do you want to continue predicting with live video? (Y/N)").upper()
    answer = input(
        "Do you want to continue predicting with live video? (Y/N)").upper()

    if answer == "Y":
        pass
    else:
        exit(1)
else:
    print("There are not all .keras models in the directory. Please check it again!")


def preprocess_frame(
    frame: cv2.typing.MatLike, target_size: tuple[int, int]
) -> np.ndarray:
    """
    Process the video frame

    Parameters
    ----------
    frame : cv2.typinh.Matlike
            Video frame

    target_size : tuple[int, int]
            x and y sizes for the frame

    Returns
    -------
    result : np.ndarray
            Expanded shape of the video frame
    """
    frame = cv2.resize(frame, target_size)
    frame = frame / 255.0

    return np.expand_dims(frame, axis=0)


def classify_frame(frame: cv2.typing.MatLike) -> tuple[str, str, str, str, str]:
    """
    Classifies various features of a face detected in a video frame.

    This function processes a given video frame to predict the gende, presence
    of a beard, haircolor, presence of glasses, and nationality of the person
    int the frame

    Parameters
    ----------
    frame : cv2.typing.MatLike
        The input video frame containing the face to be classified.

    Returns
    -------
    tuple[str, str, str, str, str]
        A tuple containing the following classifications:
        - Gender: "Male" or "Female".
        - Beard: "Beard" or "No Beard".
        - Hair color: A string representing the predicted hair color.
        - Glasses: "Glasses" or "No Glasses".
        - Nationality: A string representing the predicted nationality.
    """

    # TODO: First let the face detection model detect a face on the live video
    # feed. Then extract the detected face and use this as a "input image" for
    # the other feature detection models

    frame_input = preprocess_frame(frame, target_size)
    # beard_input = preprocess_frame(frame, target_size)
    # haircolor_input = preprocess_frame(frame, target_size)

    gender_prediction = models["Gender_classification.keras"].predict(
        frame_input)
    beard_prediction = models["Beard_classification.keras"].predict(
        frame_input)
    haircolor_prediction = models["Haircolor_classification.keras"].predict(
        frame_input)
    nation_prediction = models["Nation_classification.keras"].predict(
        frame_input)
    glasses_prediction = models["Glasses_classification.keras"].predict(
        frame_input)

    gender = "Male" if gender_prediction[0][0] > 0.5 else "Female"
    beard = "No Beard" if beard_prediction[0][0] > 0.5 else "Beard"
    haircolor = haircolor_labels[np.argmax(haircolor_prediction)]
    glasses = "No Glasses" if glasses_prediction[0][0] > 0.5 else "Glasses"
    nation = nation_labels[np.argmax(nation_prediction)]

    return gender, beard, haircolor, glasses, nation


# Try either 0 or 1, it depends on your OS
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Error: Camera couldn't start")
    exit(1)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: There is no frame")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    gender, beard, haircolor, glasses, nation = classify_frame(rgb_frame)

    text = f"Gender: {gender}, Beard: {beard}, Haircolor: {haircolor}, Glasses: {glasses}, Nation: {nation}"
    cv2.putText(frame, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Camera Window", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
