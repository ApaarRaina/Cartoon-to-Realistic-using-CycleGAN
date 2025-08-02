import os
import random
import shutil
from tqdm import tqdm

# Paths
source_dir = 'frontal_faces'
trainB_dir = 'trainB'
testB_dir = 'testB'

# Ensure output directories exist
os.makedirs(trainB_dir, exist_ok=True)
os.makedirs(testB_dir, exist_ok=True)

# List and shuffle images
all_images = os.listdir(source_dir)
random.shuffle(all_images)

# Split into train and test
train_images = all_images[:9148]
test_images = all_images[9148:9148 + 901]

# Copy files to respective directories
for img_name in tqdm(train_images):
    shutil.copy(os.path.join(source_dir, img_name), os.path.join(trainB_dir, img_name))

for img_name in tqdm(test_images):
    shutil.copy(os.path.join(source_dir, img_name), os.path.join(testB_dir, img_name))

print(f"Copied {len(train_images)} images to trainB/")
print(f"Copied {len(test_images)} images to testB/")
