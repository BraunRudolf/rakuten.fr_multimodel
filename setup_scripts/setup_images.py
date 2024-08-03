import os
import shutil
import zipfile

from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

IMAGE_ZIP = os.getenv("IMAGE_ZIP")
IMAGE_FOLDER = os.getenv("IMAGE_FOLDER")
IMAGE_TRAIN_FOLDER = os.getenv("IMAGE_TRAIN_FOLDER")


def unzip_image(zip_path):
    with zipfile.ZipFile(zip_path, "r") as zip:
        # Get a list of all members (files and directories) in the zip file
        members = zip.infolist()

        # Iterate over members with a progress bar
        for member in tqdm(members, desc="Extracting files"):
            zip.extract(member, "./")


def remove_folder_from_dir(folder_path):
    for file in os.listdir(folder_path):
        if os.path.isdir(os.path.join(folder_path, file)):
            shutil.rmtree(os.path.join(folder_path, file))


def move_content_to_parent(folder_path):
    parent_dir = os.path.dirname(folder_path)
    for file in os.listdir(folder_path):
        shutil.move(os.path.join(folder_path, file), parent_dir)


def main():
    unzip_image(IMAGE_ZIP)
    move_content_to_parent(os.path.join(IMAGE_FOLDER, IMAGE_TRAIN_FOLDER))
    remove_folder_from_dir(IMAGE_FOLDER)


if __name__ == "__main__":
    main()
