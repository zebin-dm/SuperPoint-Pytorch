#-*-coding:utf8-*-
import os
from loguru import logger
import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import cv2
import yaml
import shutil
from dataset.synthetic_shapes import SyntheticShapes
# from model.magic_point import MagicPoint
# from model.magic_point_dm import MagicPoint
from model.magic_point_v1 import MagicPoint
from torch.utils.data import DataLoader


with open('./config/magic_point_syn_train.yaml', 'r', encoding='utf8') as fin:
    config = yaml.safe_load(fin)

device = 'cpu' #'cuda:2' if torch.cuda.is_available() else 'cpu'
dataset_ = SyntheticShapes(config['data'], task='training', device=device)
dataloader_ = DataLoader(dataset_, batch_size=1, shuffle=False, collate_fn=dataset_.batch_collator)
save_path = os.path.join(config["solver"]["save_dir"], "visualization")

if os.path.exists(save_path):
    shutil.rmtree(save_path)
os.makedirs(save_path)

logger.info("the visual path: {}".format(save_path))

net = MagicPoint(config['model'])

net.load_state_dict(torch.load(os.path.join(config["solver"]["save_dir"], "mg_syn_0039_0.112.pth")))
net.to(device).eval()

coco_path = "/home/dm/work/04.dataset/coco/test2017"
image_list = os.listdir(coco_path)



with torch.no_grad():
    # for i, data in tqdm(enumerate(dataloader_)):
    for i, img_name in enumerate(image_list):
        img_file = os.path.join(coco_path, img_name)
        rgb_img = cv2.imread(img_file)
        gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
        img_tensor = torch.as_tensor(gray_img, device=device, dtype=torch.float32)
        img_tensor = img_tensor.unsqueeze(dim=0).unsqueeze(dim=0) / 255.0
        # ret = net(img_tensor)
        x_s4, x_s8, logits, prob_no_nms, prob_nms = net(img_tensor)
        ##debug
        if i > 10:
            break
        # warp_img = (data['raw']['img'] * 255).cpu().numpy().squeeze().astype(np.int).astype(np.uint8)
        # warp_img = cv2.merge((warp_img, warp_img, warp_img))
        # prob = ret['prob_nms'].cpu().numpy().squeeze()
        prob = prob_nms.cpu().numpy().squeeze()
        keypoints = np.where(prob > 0.015)
        keypoints = np.stack(keypoints).T
        for kp in keypoints:
            cv2.circle(rgb_img, (int(kp[1]), int(kp[0])), radius=3, color=(0, 255, 0))
        save_img_file = os.path.join(save_path, "{:0>4d}.jpg".format(i))
        cv2.imwrite(save_img_file, rgb_img)
        # plt.imshow(warp_img)
        # plt.show()


print('Done')