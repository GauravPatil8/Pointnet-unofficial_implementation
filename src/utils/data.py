import os
import zipfile
import urllib.request


def unzip(extract_path, file_path):
    """ Extracts zip file to given location"""

    if not os.path.exists(extract_path):
        os.makedirs(extract_path)

    try:
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)

    except zipfile.BadZipFile:
        print(f"BadZipFile: The file at {file_path} is not a zip file or it is corrupted.")

    except Exception as e:
        print(f"Unexpected error while extracting ZIP: {e}")


def download_data(url, download_path, extract_path, file_name):
    """Downloads and extracts data from the web"""

    if not os.path.exists(download_path):
        os.makedirs(download_path)

    zip_path = os.path.join(download_path, file_name) # "modelnet.zip"

    try:
        if not os.path.exists(zip_path):
            print(f"Downloading Data: {url} at location: {zip_path}")
            urllib.request.urlretrieve(url, zip_path)
            print("Downloading finished")
        unzip(extract_path = extract_path, file_path = zip_path)

    except urllib.error.HTTPError as e:
        print(f"HTTP Error: {e.code} - {e.reason} for URL: {url}")

    except urllib.error.URLError as e:
        print(f"URL Error: {e.reason} for URL: {url}")

    except Exception as e:
        print(e)
