# These are the imports, that are necessary for the project
import glob
import zipfile
import os

import numba as nb
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from pathlib import Path
from PIL import Image
from sklearn.metrics import f1_score
from tensorflow.keras import regularizers, initializers, Sequential
from tensorflow.keras.layers import Conv2D, Dropout, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator


train_data_path = 'Train'
test_data_path = 'Test'
validation_data_path = 'Validation'





# Hyperparamter
batch_size = 24
learning_rate = 0.0001
num_images = 16


# Read train data from directory and augment data
train_datagen = ImageDataGenerator( # Data Augumentation for test data
    rotation_range=30,
    shear_range=0.3,
    zoom_range=0.3,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    rescale=1./255,
)

train_gen=train_datagen.flow_from_directory(
    train_data_path,
    target_size=(250,250),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=True
)

# Read validation data from file and augment data
valid_datagen = ImageDataGenerator(rescale=1./255)

valid_gen=valid_datagen.flow_from_directory(
    validation_data_path,
    target_size=(250,250),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=True
)


# Read test data from file and augment data
valid_datagen = ImageDataGenerator(rescale=1./255)

test_gen=valid_datagen.flow_from_directory(
    test_data_path,
    target_size=(250,250),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=True
)





x_batch, y_batch = next(train_gen)

fig, axes = plt.subplots(4, 4, figsize=(12, 10))

for i in range(num_images):
    ax = axes[i // 4, i % 4]
    ax.imshow(x_batch[i])
    label = "Female" if y_batch[i] == 0 else "Male"
    ax.set_title(label)
    ax.axis('off')

plt.tight_layout()
plt.show()






train_data_path_male = Path(train_data_path) / "Male"
train_data_path_female = Path(train_data_path) / "Female"

size_of_male_data = len([file_or_dir for file_or_dir in os.listdir(train_data_path_male)
                         if os.path.isfile(train_data_path_male / file_or_dir) and file_or_dir.endswith(".jpg")])
size_of_female_data = len([file_or_dir for file_or_dir in os.listdir(train_data_path_female)
                           if os.path.isfile(train_data_path_female / file_or_dir) and file_or_dir.endswith(".jpg")])

print(size_of_female_data, size_of_male_data)









kernel_s = (3, 3)
epochs, steps_per_epoch = 30, 100
early_stopping = EarlyStopping(monitor='val_loss', patience=10)  # Early stopping if model can't get better in training
class_weight = {"male": 1.0, "female": size_of_male_data / size_of_male_data}

model = Sequential([
    Conv2D(32 ,kernel_s ,activation='relu',input_shape=(250,250,3), kernel_regularizer=regularizers.l2(0.001),padding="VALID"),
    MaxPooling2D((2,2)),
    
    Conv2D(64,kernel_s,activation='relu'),
    MaxPooling2D((2,2)),

    Conv2D(64,kernel_s,activation='relu'),
    MaxPooling2D((2,2)),
    
    Conv2D(128,kernel_s,activation='relu'),
    MaxPooling2D((2,2)),

    Conv2D(128,kernel_s,activation='relu'),
    MaxPooling2D((2,2)),

    # Last layer decise -> preparation for this = some layers before
    Flatten(),
    Dense(256, activation='relu'),
    # Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Mögliche Verbersserungen:
#    1. Filtergröße ändern
#    2. Mehr Layer hinzufügen
#    3. MaxPooling2D entfernen bzw. nach hinten schieben = Conv2D (konvolutionale Layer) hintereinander reihen -> könnte das Modell komplexer machen = rechenintensiver
#    -> Schlechter geworden
#    -> Mehr Epochen als 30 sind zwar auch nicht schlecht, aber die Verbesserung erhöht sich nicht viel => sehr wahrscheinlich Overfitting vorhanden
#    4. Unit, also Ausgabegröße, von letztem Layer (Dense) erhöhen (> 10): 5 % besser geworden -> warum wird die Unit-size so klein gewählt???


model.compile(
    optimizer='rmsprop',
    loss='binary_crossentropy',
    metrics=['acc']
)


history = model.fit(
        train_gen, 
        validation_data=valid_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        callbacks=[early_stopping]
    )

# Plot history
plt.plot(history.history['acc'], label='accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='loss')
plt.legend()





# Test model
x_batch, y_batch = next(test_gen)

predictions = model.predict(x_batch)
print(f"Predictions {predictions}")

predicted_labels = (predictions > 0.5).astype(int).flatten()

num_images = 16
fig, axes = plt.subplots(4, 4, figsize=(12, 10))

class_names = ["Female", "Male"]

for i in range(num_images):
    ax = axes[i // 4, i % 4]
    ax.imshow(x_batch[i])
    
    true_label = class_names[int(y_batch[i])]
    predicted_label = class_names[predicted_labels[i]]
    
    ax.set_title(f"Original: {true_label}\nPredicted: {predicted_label}")
    ax.axis('off')

plt.tight_layout()
plt.show()

# Save model
model.evaluate(valid_gen)
model.save("Gender_classification.keras")






# Load model
if os.path.isfile('Gender_classification.keras'):
    model = tf.keras.models.load_model('Gender_classification.keras')
else:
    print("Gender classification model could not be loaded")


# Probe model
for i, image in enumerate(images):
    img_array = edit_image(image)
    
    prediction = model.predict(img_array)
    predicted_labels = (prediction > 0.5).astype(int).flatten()
    predicted_label = class_names[predicted_labels[0]]

    plt.imshow(load_img(image))
    plt.title(f"Prediction: {predicted_label} {prediction}")
    plt.axis("off")
    plt.show()