import nbformat
import subprocess
import os
import zipfile

from pathlib import Path
from typing import Dict
from nbconvert.preprocessors import ExecutePreprocessor

def run_notebook(notebook_path: str) -> None:
    """
    Executes a Jupyter notebook at the specified path.
    This function reads a Jupyter notebook file, executes all its cells.
    If an error occurs during execution, it prints the error message and re-raises the exception.
    
    Parameters
    ----------
    notebook_path : str
        The file path to the Jupyter notebook to be executed.
        
    Raises
    ------
    Exception
        If an error occurs during the execution of the notebook.
    """
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    
    try:
        ep.preprocess(nb, {'metadata': {'path': '.'}})
        print(f"Notebook: {notebook_path} was executed successfully!")
    except Exception as e:
        print(f"Error ocured while trying to exeucte the notebook: {e}")
        raise


def extract_dataset(output_path: str, extract_path: str) -> None:
    """
    Extracts a dataset from a zip file if it has not already been extracted.
    Checks whether the specified extraction directory exists and is non-empty.
    If not, extracts the contents of the zip file at `output_path` into `extract_path`.
    
    Parameters
    ----------
    output_path : str
        The file path to the zip file containing the dataset.
    extract_path : str
        The directory path where the dataset should be extracted.
        
    Returns
    -------
    None
    """
    
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
    """
    Downloads datasets from specified URLs to given output paths.

    For each URL and output path pair in the input dictionary, this function checks if the dataset already exists at the output path. If not, it downloads the dataset using `curl`. If the download is successful, a confirmation message is printed. If the file already exists, the download is skipped.

    Parameters
    ----------
    download_urls_with_output_path : Dict[str, str]
        A dictionary mapping download URLs (str) to their corresponding output file paths (str).

    Returns
    -------
    None

    Raises
    ------
    Prints error messages if the download fails or if an unknown error occurs.
    """
    
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
    
    # Downloads the datasets
    download_datasets(download_urls_with_output_path)
    
    # Extracts the datasets
    for _, output_path in download_urls_with_output_path.items():
        extract_path = Path(output_path).with_suffix('')
        extract_dataset(output_path, extract_path)
    
    path_to_notebooks = ["gender/ProcessGenderImages.ipynb", "beard/ProcessBeardImages.ipynb", "glasses/ProcessGlassesImages.ipynb", "haircolor/ProcessHaircolorImages.ipynb", "nation/ProcessNationImages.ipynb"]

    for notebook in path_to_notebooks:
        print(f"Executing notebook: {notebook}")
        run_notebook(notebook_path=notebook)
    
   