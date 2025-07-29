import os
import shutil

def flatten_feather_files(main_folder):
    for root, dirs, files in os.walk(main_folder, topdown=False):
        for name in files:
            if name.endswith('.feather'):
                src = os.path.join(root, name)
                dst = os.path.join(main_folder, name)
                shutil.move(src, dst)
        for name in dirs:
            dir_path = os.path.join(root, name)
            if dir_path != main_folder and os.path.isdir(dir_path):
                shutil.rmtree(dir_path)

# USAGE:
main_folder = 'output_ultrafast'  # Change to your folder path
flatten_feather_files(main_folder)

