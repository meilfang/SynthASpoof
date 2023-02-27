import os
import cv2
import numpy as np
from tqdm import tqdm
import argparse

from facenet_pytorch import MTCNN

mtcnn = MTCNN(select_largest=True, min_face_size=64, post_process=False, device='cuda:0')

def crop_images(in_folder, out_folder):
    print('Imges are in :', in_folder)
    os.makedirs(out_folder, exist_ok=True)
    skipped_imgs = []
    no_used_images = []
    img_names = os.listdir(in_folder)
    for img_name in tqdm(img_names):
        filepath = os.path.join(in_folder, img_name)
        if os.path.isfile(os.path.join(out_folder, img_name)):
            continue
        else:
            img = cv2.imread(filepath)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if img is None:
                continue
            boxes, probs = mtcnn.detect(img)

            if boxes is None:
                skipped_imgs.append(img_name)
                continue

            x1, y1, x2, y2 = boxes[0]
            x1, y1 = max(x1, 0), max(y1, 0)
            crop_img = img[int(y1):int(y2), int(x1):int(x2)]

            try:
                crop_img = cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(out_folder, img_name), crop_img)
            except:
                skipped_imgs.append(img_name)
                pass

    print(skipped_imgs)
    print(f"Images with no Face: {len(skipped_imgs)}")

def main():
    parser = argparse.ArgumentParser(description='Face detection and crop')
    parser.add_argument('--in_folder', type=str, default="./in", help='folder with images')
    parser.add_argument('--out_folder', type=str, default="./out", help="folder to save aligned images")

    args = parser.parse_args()
    crop_images(args.in_folder, args.out_folder)

if __name__ == "__main__":
    main()
