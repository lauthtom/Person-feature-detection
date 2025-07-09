import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from keras.preprocessing.image import img_to_array, load_img
from PIL import Image
from test_on_camera import load_models
from Face_detection.face_detection import detect_face


def classify_frame(image: np.ndarray) -> tuple[str, str, str, str, str]:
    """
    Classifies various attributes from an input image using the trained models.

    Parameters
    ----------
    image : np.ndarray
        The input image as a NumPy array to be classified.

    Returns
    -------
    tuple of str
        A tuple containing the following predicted attributes:
            - gender (str): Predicted gender ("Male" or "Female").
            - gender_score (float): Accuracy for gender prediction.
            - beard (str): Predicted beard status ("No Beard" or "Beard").
            - beard_score (float): Accuracy for beard prediction.
            - haircolor (str): Predicted hair color label.
            - glasses (str): Predicted glasses status ("No Glasses" or "Glasses").
            - glasses_score (float): Accuracy for glasses prediction.
            - nation (str): Predicted nationality label.

    Notes
    -----
    This function assumes that the required models and label lists
    (`models`, `haircolor_labels`, `nation_labels`) are defined in the global scope.
    """
    gender_prediction = models["Gender_classification"].predict(
        image)
    beard_prediction = models["Beard_classification"].predict(
        image)
    haircolor_prediction = models["Haircolor_classification"].predict(
        image)
    nation_prediction = models["Nation_classification"].predict(
        image)
    glasses_prediction = models["Glasses_classification"].predict(
        image)

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
        futures = [executor.submit(load_models, name, path_to_models)
                   for name in model_names]
        for future in as_completed(futures):
            name, model = future.result()
            models[name] = model

    print("All models loaded successfully!")

    answer = input(
        "Do you want to continue predicting with static images? (Y/N)").upper()

    if answer == "Y":
        images_dir = os.listdir("Images/")
        results = []

        for image in images_dir:
            if image.endswith(".jpg"):
                image_path = os.path.join("Images/", image)
                img = load_img(f"Images/{image}", target_size=target_size)
                img_array = img_to_array(img)
                img_array = img_array / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                img_detected_face = detect_face(
                    image_path=image_path, padding=0.2)

                if img_detected_face is not None:
                    if isinstance(img_detected_face, np.ndarray):
                        img_detected_face = Image.fromarray(img_detected_face)

                    img_detected_face = img_detected_face.resize(target_size)
                    img_array = img_to_array(img_detected_face)
                    img_array = img_array / 255.0
                    img_array = np.expand_dims(img_array, axis=0)

                    gender, gender_acc, beard, beard_acc, haircolor, glasses, glasses_acc, nation = classify_frame(
                        img_array)

                    plt.figure(figsize=(5, 5))
                    plt.imshow(img)
                    plt.axis("off")
                    plt.title(
                        f"Gender: {gender} ({gender_acc*100:.1f}%)\n"
                        f"Beard: {beard} ({beard_acc*100:.1f}%)\n"
                        f"Glasses: {glasses} ({glasses_acc*100:.1f}%)\n"
                        f"Haircolor: {haircolor}\n"
                        f"Nation: {nation}",
                        fontsize=14,
                        loc="center"
                    )
                    plt.tight_layout()
                    plt.show()
            else:
                print(f"Skipped {image}, because it's not a image")
    else:
        exit(1)
