import cv2
import os

import numpy as np
import tensorflow as tf

from concurrent.futures import ThreadPoolExecutor, as_completed
from Face_detection.face_detection import detect_face_from_frame
from keras.models import Sequential


def load_models(name: str, path_to_models: str) -> tuple[str, Sequential]:
    """
    Loads a TensorFlow Keras model from the specified model name.

    Parameters
    ----------
    name : str
        The name of the model to load.
    path_to_models : str
        The path of the model to load.

    Returns
    -------
    tuple
        A tuple containing the model name and the loaded Keras model instance.

    """
    model_path = os.path.join(path_to_models, name)
    print(f"Loading the {name} model")
    model = tf.keras.models.load_model(model_path)
    print(f"Successfully loaded the {name} model")

    return name, model


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

    frame_input = preprocess_frame(frame, target_size)

    gender_prediction = models["Gender_classification"].predict(
        frame_input)
    beard_prediction = models["Beard_classification"].predict(
        frame_input)
    haircolor_prediction = models["Haircolor_classification"].predict(
        frame_input)
    nation_prediction = models["Nation_classification"].predict(
        frame_input)
    glasses_prediction = models["Glasses_classification"].predict(
        frame_input)

    gender = "Male" if gender_prediction[0][0] > 0.5 else "Female"
    beard = "No Beard" if beard_prediction[0][0] > 0.5 else "Beard"
    haircolor = haircolor_labels[np.argmax(haircolor_prediction)]
    glasses = "No Glasses" if glasses_prediction[0][0] > 0.5 else "Glasses"
    nation = nation_labels[np.argmax(nation_prediction)]

    return gender, gender_prediction[0][0], beard, beard_prediction[0][0], haircolor, glasses, glasses_prediction[0][0], nation


if __name__ == "__main__":

    # All the necessary variables
    haircolor_labels = ["Black hair", "Blond hair", "Brown hair", "Gray hair"]
    nation_labels = ["Asian", "Black", "Indian", "Others", "White"]
    path_to_models = "saved_models/"
    target_size = (224, 224)
    models = {}
    model_names = [
        "Gender_classification",
        "Beard_classification",
        "Haircolor_classification",
        "Nation_classification",
        "Glasses_classification",
    ]

    # Parallel loading of the different models
    with ThreadPoolExecutor(max_workers=len(model_names)) as executor:
        futures = [executor.submit(load_models, name, path_to_models) for name in model_names]
        for future in as_completed(futures):
            name, model = future.result()
            models[name] = model

    print("All models loaded successfully!")

    answer = input(
        "Do you want to continue predicting with live video? (Y/N)").upper()

    if answer == "Y":

        # Try either 0 or 1, it depends on your OS
        cap = cv2.VideoCapture(1)
        cv2.namedWindow("Camera Window", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Camera Window", 640, 480)

        if not cap.isOpened():
            print("Error: Camera couldn't start")
            exit(1)

        while True:
            ret, frame = cap.read()

            if not ret:
                print("Error: There is no frame")
                break

            detected_face, face_box = detect_face_from_frame(
                frame)
            # rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if detected_face is not None:
                gender, gender_accuracy, beard, beard_accuracy, haircolor, glasses, glasses_accuracy, nation = classify_frame(
                    detected_face)

                x, y, w, h = face_box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                lines = [
                    f"Gender: {gender}, Accuracy: {gender_accuracy:.2f}%",
                    f"Beard: {beard}, Accuracy: {beard_accuracy:.2f}%",
                    f"Haircolor: {haircolor}, Glasses: {glasses}, Accuracy: {glasses_accuracy:.1f}%",
                    f"Nation: {nation}"
                ]

                y0 = 30
                dy = 35

                for i, line in enumerate(lines):
                    y = y0 + i * dy
                    cv2.putText(frame, line, (10, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "No face detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # gender, gender_accuracy, beard, beard_accuracy, haircolor, glasses, glasses_accuracy, nation = classify_frame(
            # rgb_frame)
            # gender, gender_accuracy = classify_frame(rgb_frame)

            # text = f"Gender: {gender}, Accuracy: {gender_accuracy:.1f}%, Beard: {beard}, Accuracy: {beard_accuracy:.1f}%, Haircolor: {haircolor}, Glasses: {glasses}, Accuracy: {glasses_accuracy:.1f}%, Nation: {nation}"
            # cv2.putText(frame, text, (10, 30),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # lines = [
            #     f"Gender: {gender}, Accuracy: {gender_accuracy:.1f}%",
                # f"Beard: {beard}, Accuracy: {beard_accuracy:.1f}%",
                # f"Haircolor: {haircolor}, Glasses: {glasses}, Accuracy: {glasses_accuracy:.1f}%",
                # f"Nation: {nation}"
            # ]

            # y0 = 30
            # dy = 35

            # for i, line in enumerate(lines):
            #     y = y0 + i * dy
            #     cv2.putText(frame, line, (10, y),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Camera Window", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
    else:
        exit(1)
