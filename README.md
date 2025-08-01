# Person Feature Detection

This project focuses on detecting and classifying human face attributes in images and videos using machine learning models.

## 📦 Project Setup

### 1. Clone the Repository

Start by cloning the repository. Make sure to choose the target directory carefully, as it will be needed in later steps.

```bash
git clone https://github.com/lauthtom/Person-feature-detection.git
cd Person-feature-detection
```

---

### 2. Create a Virtual Environment (Recommended: Anaconda)

Create a new virtual environment using Anaconda with Python 3.11.7:

```bash
conda create -n <environment_name> python=3.11.7
conda activate <environment_name>
```

---

### 3. Ensure `pip` is Installed

Check whether `pip` is available:

```bash
pip --version
```

If an error appears, install `pip` with:

```bash
conda install pip
```

---

### 4. Install Dependencies

Install all required Python packages listed in the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

---

## 📂 Prepare the Datasets

Run the following script to prepare and split the datasets:

```bash
python prepare_and_split_datasets.py -p <path_of_the_project>
```

Replace `<path_of_the_project>` with the absolute path to your project directory.

---

## 🧠 Train the Models

Navigate to the appropriate subdirectory and open the Jupyter Notebook (e.g., `gender/GenderClassification.ipynb`) using Jupyter Notebook or JupyterLab.

Execute the cells to follow and run the entire training process.

Once training is complete, the models will be saved in the `Models/` directory.

---

## 🎥 Live Testing with Camera (Optional)

To test the trained models in a live video feed from your webcam, run:

```bash
python text_on_camera.py
```

---

## 👤 Author

This project is developed and maintained exclusively by:

**Tom Lauth**  
© 2025 — All rights reserved unless otherwise stated.
