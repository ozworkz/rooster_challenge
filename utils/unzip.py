import os
import zipfile
from pathlib import Path

def unzip(zip_file_path, root=os.path.expanduser('./')):
    folders = []
    with zipfile.ZipFile(zip_file_path) as zf:
        zf.extractall(root)
        for name in zf.namelist():
            folder = Path(name).parts[0]
            if folder not in folders:
                folders.append(folder)

    folders = folders[0] if len(folders) == 1 else tuple(folders)
    return folders