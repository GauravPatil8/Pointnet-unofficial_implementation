import os
import zipfile
import logging
import trimesh
import numpy as np
import urllib.request

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
    """Downloads and extracts data from the web"""

    logger = logging.getLogger(__name__)
    
    if not os.path.exists(download_path):
        os.makedirs(download_path)

    zip_path = os.path.join(download_path, file_name) # "modelnet.zip"

    try:
        if not os.path.exists(zip_path):
            logger.info(f"Downloading Data: {url} at location: {zip_path}")
            urllib.request.urlretrieve(url, zip_path)
            logger.info("Downloading finished")
        unzip(extract_path = extract_path, file_path = zip_path)

    except urllib.error.HTTPError as e:
        logger.error(f"HTTP Error: {e.code} - {e.reason} for URL: {url}")

    except urllib.error.URLError as e:
        logger.error(f"URL Error: {e.reason} for URL: {url}")

    except Exception as e:
        logger.error(e)

def read_off(file_path):
    """Reads .off file type and returns vertices in nparray."""
    logger = logging.getLogger(__name__)
    try:
        mesh = trimesh.load(file_path, file_type='off')
        verts = mesh.vertices
        return np.array(verts, dtype=np.float32)
    except Exception as e:
        logger.error(f"Failed to load OFF file {file_path}: {e}")
        raise ValueError(f"Failed to load OFF file {file_path}: {e}")