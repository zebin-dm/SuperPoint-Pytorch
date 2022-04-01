# -*-coding:utf8-*-
import os
import argparse
import torch
import pprint
from torch.optim import lr_scheduler
import numpy as np
import yaml
from loguru import logger
from dataset.coco import COCODataset
from dataset.synthetic_shapes import SyntheticShapes
from torch.utils.data import DataLoader
from model.magic_point_v3 import MagicPoint
from solver.loss import loss_func


def train_eval(model, dataloader, config):
    base_lr = config['solver']['base_lr'] * config['solver']["train_batch_size"]/64
    optimizer = torch.optim.AdamW(params=model.parameters(),
                                  lr=base_lr,
                                  weight_decay=config['solver']['weight_decay'],
                                  amsgrad=True)

    train_epoches = int(config['solver']['epoch'] * config['solver']["train_batch_size"]/64)
    try:
        # start training
        for epoch in range(train_epoches):
            model.train()
            mean_loss = []
            for i, data in enumerate(dataloader['train']):
                prob, desc, prob_warp, desc_warp = None, None, None, None
                if config['model']['name'] == 'magicpoint' and config['data']['name'] == 'coco':
                    data['raw'] = data['warp']
                    data['warp'] = None
                img_data = data['raw']["img"]
                prob = model(img_data)

                ##loss
                loss = loss_func(config['solver'], data, prob, desc,
                                 prob_warp, desc_warp, device)

                mean_loss.append(loss.item())
                #reset
                model.zero_grad()
                loss.backward()
                optimizer.step()

                if (i % 100 == 0 and i != 0):
                    print("Epoch [{}/{}], Step [{}/{}], LR [{}], Loss: {:.3f}".format(
                        epoch, train_epoches, i, len(dataloader['train']),
                        optimizer.state_dict()['param_groups'][0]['lr'], np.mean(mean_loss)))
                    mean_loss = []

            # scheduler.step()
            ##do evaluation
            if (epoch % 1 ==0):
                model.eval()
                eval_loss = do_eval(model, dataloader['test'], config, device)
                save_path = os.path.join(config['solver']['save_dir'],
                                         config['solver']['model_name'] + 
                                         '_{:0>4d}_{}.pth'.format(epoch, round(eval_loss, 3)))
                torch.save(model.state_dict(), save_path)
                print('Epoch [{}/{}], Loss: {:.3f}, EvalLoss: {:.3f}, Checkpoint saved to {}'.format(
                    epoch, train_epoches, np.mean(mean_loss), eval_loss, save_path))
                mean_loss = []

    except KeyboardInterrupt:
        torch.save(model.state_dict(), "./export/key_interrupt_model.pth")

@torch.no_grad()
def do_eval(model, dataloader, config, device):
    mean_loss = []
    truncate_n = max(int(0.1 * len(dataloader)), 100)  # 0.1 of test dataset for eval

    for ind, data in enumerate(dataloader):
        if ind>truncate_n:
            break
        prob, desc, prob_warp, desc_warp = None, None, None, None
        if config['model']['name'] == 'magicpoint' and config['data']['name'] == 'coco':
            data['raw'] = data['warp']
            data['warp'] = None

        img_data = data['raw']["img"]
        prob = model(img_data)
        # compute loss
        loss = loss_func(config['solver'], data, prob, desc,
                         prob_warp, desc_warp, device)

        mean_loss.append(loss.item())
    mean_loss = np.mean(mean_loss)

    return mean_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    config_file = args.config
    assert (os.path.exists(config_file))

    with open(config_file, 'r') as fin:
        config = yaml.safe_load(fin)

    if not os.path.exists(config['solver']['save_dir']):
        os.makedirs(config['solver']['save_dir'])

    device = torch.device(config['solver']['device'])

    # Make Dataloader
    data_loaders = None
    if config['data']['name'] == 'coco':
        datasets = {k: COCODataset(config['data'], is_train=True if k == 'train' else False, device=device)
                    for k in ['test', 'train']}
        data_loaders = {k: DataLoader(datasets[k],
                                      config['solver']['{}_batch_size'.format(k)],
                                      collate_fn=datasets[k].batch_collator,
                                      shuffle=True) for k in ['train', 'test']}
    elif config['data']['name'] == 'synthetic':
        datasets = {'train': SyntheticShapes(config['data'], task=['training', 'validation'], device=device),
                    'test': SyntheticShapes(config['data'], task=['test', ], device=device)}
        data_loaders = {'train': DataLoader(datasets['train'], batch_size=config['solver']['train_batch_size'], shuffle=True,
                                            num_workers=4,
                                            collate_fn=datasets['train'].batch_collator),
                        'test': DataLoader(datasets['test'], batch_size=config['solver']['test_batch_size'], shuffle=False,
                                           num_workers=4,
                                           collate_fn=datasets['test'].batch_collator)}

    model = MagicPoint(nms=config['model']["nms"], bb_name=config['model']['bb_name'])

    # Load Pretrained Model
    if os.path.exists(config['model']['pretrained_model']):
        pre_model_dict = torch.load(config['model']['pretrained_model'])
        logger.info("load pretrained model: {}".format(config['model']['pretrained_model']))
        model.load_state_dict(pre_model_dict)
    model.to(device)
    pprint.pprint(config)
    train_eval(model, data_loaders, config)
    print('Done')
