import os

root_dir = '/media/slow_drive/VGGFace2/train_cropped'

for root, _, files in os.walk(root_dir):
    for file_name in files:
        if os.path.getsize(os.path.join(root, file_name)) == 0:
            os.remove(os.path.join(root, file_name))
            #print(os.path.join(root, file_name))