import os

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tqdm import tqdm

model_names = [
    "Gender_classification.keras",
    "Beard_classification.keras",
    "Haircolor_classification.keras",
    "Nation_classification.keras",
    "Glasses_classification.keras",
]

models_dir = "Models/"

if len(os.listdir(models_dir)) == len(model_names):

    models = {}

    for model_name in model_names:
        model_path = os.path.join(models_dir, model_name)
        print(f"Trying to load the {model_name} model...")
        models[model_name] = tf.keras.models.load_model(model_path)
        print(f"Successfully loaded the {model_name} model.")

    print("All models loaded successfully!")

    answer = input("Do you want to continue predicting with live video? (Y/N)").upper()

    if answer == "Y":
        pass
    else:
        exit(1)
else:
    print("There are not all .keras models in the directory. Please check it again!")

images_dir = os.listdir("Images/")
print(models)

# All labels for each feature with more than 2 values
haircolor_labels = ["Black hair", "Blond hair", "Brown hair", "Gray hair"]
nation_labels = ["Asian", "Black", "Indian", "Others", "White"]

target_size = (224, 224)

for image in images_dir:

    if image.endswith(".jpg"):
        img = load_img(f"Images/{image}", target_size=target_size)
        img_array = img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        gender_pred = models["Gender_classification.keras"].predict(img_array)
        beard_pred = models["Beard_classification.keras"].predict(img_array)
        haircolor_pred = models["Haircolor_classification.keras"].predict(img_array)
        nation_pred = models["Nation_classification.keras"].predict(img_array)
        glasses_pred = models["Glasses_classification.keras"].predict(img_array)

        image_width = img.size[0]
        image_height = img.size[1]

        plt.figure(figsize=(9, 9))

        plt.imshow(img)
        plt.axis("off")

        x_pos = image_width / 2
        y_pos = image_height + 10

        gender = "Male" if gender_pred[0][0] > 0.5 else "Female"
        beard = "No Beard" if beard_pred[0][0] > 0.5 else "Beard"
        glasses = "No Glasses" if glasses_pred[0][0] > 0.5 else "Glasses"
        haircolor = haircolor_labels[np.argmax(haircolor_pred)]
        nation = nation_labels[np.argmax(nation_pred)]

        plt.title(
            f"Gender: {gender}\nBeard: {beard}\nGlasses: {glasses}\nHaircolor: {haircolor}\nNation: {nation}"
        )
        plt.show()
    else:
        print(f"Skipped {image}, because it's not a image")
