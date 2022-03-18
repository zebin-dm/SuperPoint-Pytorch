# -*-coding:utf8-*-
import torch
from solver.nms import box_nms
from model.modules.cnn.vgg_backbone import VGGBackboneBN,VGGBackbone
from model.modules.cnn.cnn_heads import DetectorHead
from model.effiUnet import EfficientMagic
from utils.debug_utils import AverageTimer



def simple_nms(scores, nms_radius: int):
    """ Fast Non-maximum suppression to remove nearby points """
    assert (nms_radius >= 0)

    def max_pool(x):
        return torch.nn.functional.max_pool2d(
            x, kernel_size=nms_radius * 2 + 1, stride=1, padding=nms_radius)

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    for _ in range(2):
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)


class MagicPoint(torch.nn.Module):
    """ Pytorch definition of SuperPoint Network. """

    def __init__(self, config, input_channel=1, grid_size=8, using_bn=True, device='cpu'):
        super(MagicPoint, self).__init__()
        self.nms = config['nms']
        self.det_thresh = config['det_thresh']
        self.topk = config['topk']
        bb_name = config["bb_name"]
        self.bb_name = bb_name
        if bb_name == "EfficientMagic":
            self.backbone = EfficientMagic()
            out_chs = 40
        elif bb_name == "VGGBackboneBN":
            self.backbone = VGGBackboneBN(config['backbone']['vgg'], input_channel, device=device)
            out_chs = 128
        elif bb_name == "VGGBackbone":
            self.backbone = VGGBackbone(config['backbone']['vgg'], input_channel, device=device)
            out_chs = 128
        else:
            raise ValueError("No ")

        

        self.detector_head = DetectorHead(input_channel=out_chs, grid_size=grid_size,using_bn=using_bn)
        self.average_time = AverageTimer()

    def forward(self, x):
        """ Forward pass that jointly computes unprocessed point and descriptor
        tensors.
        Input
          x: Image pytorch tensor shaped N x 1 x H x W.
        Output
          semi: Output point pytorch tensor shaped N x 65 x H/8 x W/8.
        """
        self.average_time.reset()
        if isinstance(x, dict):
            feat_map = self.backbone(x['img'])
        else:
            if self.bb_name == "EfficientMagic":
                feat_map, x_fine = self.backbone(x)
            else:
                feat_map = self.backbone(x)
        
        self.average_time.update("backbone")
        # print("feat_map shape: {}".format(feat_map.shape))
        outputs = self.detector_head(feat_map)   # N x H x W

        prob = outputs['prob']
        # print("prob.shape: {}".format(prob.shape))
        self.average_time.update("detect")

        if self.nms is not None:
            # prob = [box_nms(p.unsqueeze(dim=0),
            #                 self.nms,
            #                 min_prob=self.det_thresh,
            #                 keep_top_k=self.topk).squeeze(dim=0) for p in prob]
            # prob = torch.stack(prob)
            prob = simple_nms(prob, nms_radius=self.nms)
            outputs.setdefault('prob_nms',prob)
        self.average_time.update("nms")

        pred = prob[prob>=self.det_thresh]
        self.average_time.update("thresh")
        outputs.setdefault('pred', pred)

        return outputs




if __name__ == "__main__":
    import yaml
    import time
    config_file = "./config/magic_point_syn_train.yaml"
    with open(config_file, 'r') as fin:
        config = yaml.safe_load(fin)
    
    device=torch.device("cuda:0")
    net = MagicPoint(config['model'], device=device)
    net = net.to(device)
    net.eval()
    net.average_time.cuda = True
    in_size=[1, 1, 608, 608]
    with torch.no_grad():
        data = torch.randn(*in_size, device=device)
        net.average_time.add = False
        out = net(data)
        net.average_time.add = True

        run_time = 1000
        torch.cuda.synchronize()
        start_time = time.time()
        for idx in range(run_time):
            out = net(data)
        torch.cuda.synchronize()
        time_interval = time.time() - start_time
        print(time_interval)
        net.average_time.print()









