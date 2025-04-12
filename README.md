# Person-feature-detection
Detect features of persons: gender, hair colour, beard, nationality, glasses

### Datasets
The Datasets we used in this project are the following:
* https://www.kaggle.com/datasets/jessicali9530/celeba-dataset
* https://www.kaggle.com/datasets/jangedoo/utkface-new/data

If you downloaded the Datasets, you will have to unzip these into the Datasets/ directory. If this directory doesn't exist, you will have to create it.

### Running the models
Before you run the models, you will have to execute the ProcessImages ipython notebook in each feature directory. This will create you 0.80% Train, 0.15% Test and 0.05 Validate Data.

If you want to test the models with the camera on your computer, you can start the file 'test_on_camera.py' otherwise you can test the models on some images with the 'test_on_images.py' file.

### Keras models
If you execute the classification ipython notebooks, the models will be saved in the Models/ directory.
