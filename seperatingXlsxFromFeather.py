import os

targetFolder = 'change_output'

os.makedirs(f"{targetFolder}_xlsx", exist_ok=True)
for root, dirs, files in os.walk(targetFolder):
    for file in files:
        if file.endswith('.xlsx'):
            os.rename(f'{targetFolder}/{file}', f"{targetFolder}_xlsx/{file}")
