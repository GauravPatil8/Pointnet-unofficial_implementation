import os
import zipfile
import logging
import trimesh
import requests
import numpy as np
from tqdm import tqdm

def unzip(extract_path, file_path):
    """ Extracts zip file to given location"""

    logger = logging.getLogger(__name__)

    if not os.path.exists(extract_path):
        os.makedirs(extract_path)

    try:
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)

    except zipfile.BadZipFile:
        logger.info(f"BadZipFile: The file at {file_path} is not a zip file or it is corrupted.")

    except Exception as e:
        logger.info(f"Unexpected error while extracting ZIP: {e}")


def download_data(url, download_path, extract_path, file_name):
    """Downloads and extracts data from the web using requests"""
    
    logger = logging.getLogger(__name__)
    
    if not os.path.exists(download_path):
        os.makedirs(download_path)

    zip_path = os.path.join(download_path, file_name)

    try:
        if not os.path.exists(zip_path):
            logger.info(f"Downloading Data: {url} at location: {zip_path}")
            
            # Streaming request
            with requests.get(url, stream=True) as response:
                response.raise_for_status()
                total_size = int(response.headers.get('content-length', 0))
                block_size = 1024  # 1 KB
                with open(zip_path, 'wb') as f, tqdm(
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=file_name
                ) as progress_bar:
                    for chunk in response.iter_content(chunk_size=block_size):
                        if chunk:
                            f.write(chunk)
                            progress_bar.update(len(chunk))
            
            logger.info("Downloading finished")

        unzip(extract_path=extract_path, file_path=zip_path)

    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP Error: {e.response.status_code} - {e.response.reason} for URL: {url}")

    except requests.exceptions.ConnectionError as e:
        logger.error(f"Connection Error: {e} for URL: {url}")

    except requests.exceptions.Timeout as e:
        logger.error(f"Timeout Error: {e} for URL: {url}")

    except requests.exceptions.RequestException as e:
        logger.error(f"Request Error: {e} for URL: {url}")

    except Exception as e:
        logger.error(f"Unexpected Error: {e}")

def read_off(file_path, num_points):
    """Reads .off file type and returns vertices in nparray."""
    logger = logging.getLogger(__name__)
    try:
        mesh = trimesh.load(file_path, file_type='off')
        points, _ = trimesh.sample.sample_surface(mesh, num_points) #now samples from entire surface instead of just vertices
        return np.array(points, dtype=np.float32)
    
    except Exception as e:
        logger.error(f"Failed to load OFF file {file_path}: {e}")
        raise ValueError(f"Failed to load OFF file {file_path}: {e}")

def get_classes(data_dir):
    """Returns list classes present in the dataset."""
    classes = []
    for cls in os.listdir(data_dir):
        if os.path.isdir(os.path.join(data_dir,cls)):
            classes.append(cls)
    return classes