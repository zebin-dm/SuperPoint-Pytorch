#-*-coding:utf-8-*-
import os
import yaml
import torch
from loguru import logger
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from dataset.patch import PatchesDataset
from dataset.synthetic_shapes import SyntheticShapes
# from model.magic_point import MagicPoint
from model.magic_point_v3 import MagicPoint
# from model.magic_point_v2 import MagicPoint
from model.superpoint_bn import SuperPointBNNet


if __name__=="__main__":
    ##
    with open('./config/detection_repeatability_v1.yaml', 'r', encoding='utf8') as fin:
        config = yaml.safe_load(fin)

    output_dir = config['data']['export_dir']
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    device = 'cuda:0'
    if config['data']['name']=='synthetic':
        dataset_ = SyntheticShapes(config['data'], task='training', device=device)
    elif config['data']['name'] == 'hpatches':
        dataset_ = PatchesDataset(config['data'],device=device)

    p_dataloader = DataLoader(dataset_, batch_size=1, shuffle=False, collate_fn=dataset_.batch_collator)

    if config['model']['name'] == 'superpoint':
        net = SuperPointBNNet(config['model'], device=device, using_bn=config['model']['using_bn'])
    elif config['model']['name'] == 'magicpoint':
        net = MagicPoint(nms=4, bb_name="")
    logger.info("load pretrain: {}".format(config['model']['pretrained_model']))
    net.load_state_dict(torch.load(config['model']['pretrained_model'], map_location=device))
    net.to(device).eval()

    with torch.no_grad():
        for i, data in tqdm(enumerate(p_dataloader)):
            x_s8_1, logits_1, prob_no_nms_1, prob_nms_1 = net(data['img'])
            x_s8_2, logits_2, prob_no_nms_2, prob_nms_2 = net(data['warp_img'])
            ##
            pred = {'prob':prob_nms_1, 'warp_prob':prob_nms_2,
                    'homography': data['homography']}

            if not ('name' in data):
                pred.update(data)
            #to numpy
            pred = {k:v.cpu().numpy().squeeze() for k,v in pred.items()}
            filename = data['name'] if 'name' in data else str(i)
            filepath = os.path.join(output_dir, '{}.npz'.format(filename))
            np.savez_compressed(filepath, **pred)

    print('Done')