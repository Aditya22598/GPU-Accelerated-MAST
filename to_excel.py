import glob
import pandas as pd
import os

folder = 'output_ultrafast'  # Change to your folder path

feather_files = glob.glob(os.path.join(folder, '*.feather'))

for file in feather_files:
    df = pd.read_feather(file)
    out_file = file.replace('.feather', '.xlsx')
    df.to_excel(out_file, index=False)
    print(f"Converted {file} to {out_file}")

