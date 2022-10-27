import os
import sys 
import shutil 

SRC_PATH = "/sailhome/msun415/UCL_michael/"
DEST_PATH = "/nlp/scr/msun415/"
path = sys.argv[1]
reverse = False

if reverse:
    src_path = SRC_PATH
    SRC_PATH = DEST_PATH
    DEST_PATH = src_path

new_path = path.replace(SRC_PATH, DEST_PATH)

new_dir = os.path.dirname(new_path)

if os.path.isfile(path):

    os.makedirs(new_dir, exist_ok=True)

    print(f"moving {path} to {new_path}")

    shutil.move(path, new_path)



    