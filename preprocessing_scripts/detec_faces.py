from facenet_pytorch import MTCNN
from PIL import Image
from tqdm import tqdm
import os
import torch

mtcnn = MTCNN(keep_all=False, post_process=False, device='cuda' if torch.cuda.is_available() else 'cpu')

input_dir = "img_align_celeba"
output_dir = "faces_aligned_white_bg"
os.makedirs(output_dir, exist_ok=True)

for img_name in tqdm(os.listdir(input_dir)):
    img_path = os.path.join(input_dir, img_name)
    try:
        img = Image.open(img_path).convert('RGB')
        box = mtcnn.detect(img)[0]

        if box is not None:
            x1, y1, x2, y2 = [int(b) for b in box[0]]
            face_crop = img.crop((x1, y1, x2, y2))
            face_crop = face_crop.resize((256, 256))

            # Create white background
            white_bg = Image.new('RGB', (256, 256), (255, 255, 255))
            white_bg.paste(face_crop, (0, 0))

            white_bg.save(os.path.join(output_dir, img_name))
    except:
        continue
