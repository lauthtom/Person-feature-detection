import nbformat
import subprocess
import os
import zipfile

from pathlib import Path
from typing import Dict
from nbconvert.preprocessors import ExecutePreprocessor

def run_notebook(notebook_path: str) -> None:
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    
    try:
        ep.preprocess(nb, {'metadata': {'path': '.'}})
        print(f"Notebook: {notebook_path} was executed successfully!")
    except Exception as e:
        print(f"Fehler im Notebook: {e}")
        raise


def extract_dataset(output_path: str, extract_path: str) -> None:
    if not os.path.exists(extract_path) or not os.listdir(extract_path):
        print(f"Dataset {output_path} is not extracted!")

        print(f"Extracting {output_path} to {extract_path}...")
        try:
            with zipfile.ZipFile(output_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
            print("Extraction completed successfully!")
        except Exception as e:
            print(f"Error during extraction: {e}")
    else:
        print(f"Dataset {output_path} is alreay extracted!")
    

def download_datasets(download_urls_with_output_path: Dict[str, str]) -> None:
    for download_url, output_path in download_urls_with_output_path.items():
        if os.path.exists(output_path):
            print(f"Dataset already exists at {output_path} - skipping download.")
        else:
            curl_command = [
                "curl",
                "-L",
                "-o", output_path,
                download_url
            ]

            try:
                subprocess.run(curl_command, check=True)
                print(f"Successfully downloaded the dataset in the path: {output_path}")
            except subprocess.CalledProcessError as e:
                print(f"Error ocured while trying to download: {e}")
            except Exception as e:
                print(f"Unknown error: {e}")
        
if __name__ == "__main__":
    download_urls_with_output_path = {"https://www.kaggle.com/api/v1/datasets/download/jessicali9530/celeba-dataset": os.path.expanduser("~/Person-feature-detection/Datasets/celeba-dataset.zip"), "https://www.kaggle.com/api/v1/datasets/download/jangedoo/utkface-new": os.path.expanduser("~/Person-feature-detection/Datasets/utkface-dataset.zip")}
    
    download_datasets(download_urls_with_output_path)
    
    for _, output_path in download_urls_with_output_path.items():
        extract_path = Path(output_path).with_suffix('')
        extract_dataset(output_path, extract_path)
    
    # TODO: Eventuell in eine eigene Methode auslagern
    path_to_notebooks = ["gender/ProcessGenderImages.ipynb", "beard/ProcessBeardImages.ipynb", "glasses/ProcessGlassesImages.ipynb", "haircolor/ProcessHaircolorImages.ipynb", "nation/ProcessNationImages.ipynb"]

    # TODO: Execute all the notebooks
    for notebook in path_to_notebooks:
        print(f"Executing notebook: {notebook}")
        run_notebook(notebook_path=notebook)
    
   