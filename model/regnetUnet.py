import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import regnet


class conv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size, stride):
        super(conv, self).__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(num_in_layers,
                              num_out_layers,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=(self.kernel_size - 1) // 2)
        self.bn = nn.BatchNorm2d(num_out_layers)

    def forward(self, x):
        return F.elu(self.bn(self.conv(x)), inplace=True)


class upconv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size, scale):
        super(upconv, self).__init__()
        self.scale = scale
        self.conv = conv(num_in_layers, num_out_layers, kernel_size, 1)

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=self.scale, align_corners=True, mode='bilinear')
        return self.conv(x)


class RegNetMagic(nn.Module):
    def __init__(self, encoder='b0', pretrained=True, coarse_out_ch=128, fine_out_ch=128):
        super(RegNetMagic, self).__init__()
        filters = [24, 40, 112]
        regnet_ins = regnet.regnet_y_1_6gf(pretrained=True)
        print(len(regnet_ins.trunk_output))
        self.first_conv = regnet_ins.stem
        self.trunk_output = regnet_ins.trunk_output[:3]
        # if encoder == "b1":
        #     efficient_ins = efficientnet.efficientnet_b1(pretrained=pretrained)

        # self.layer2_3 = efficient_ins.features[1:3]   # H/4
        # self.layer4 = efficient_ins.features[3]      # H/8
        # self.layer5_6 = efficient_ins.features[4:6]  # H/16

        # self.upconv3 = upconv(filters[2], filters[1], 3, 2)
        # self.iconv3 = conv(filters[1] + filters[1], filters[1], 3, 1)
        # self.upconv2 = upconv(filters[1], filters[0], 3, 2)
        # self.iconv2 = conv(filters[0] + filters[0], filters[0] + filters[0], 3, 1)
        # # fine-level conv
        # self.conv_fine = conv(filters[0] + filters[0], fine_out_ch, 1, 1)

    def forward(self, x):
        # print(x.shape)
        out = self.first_conv(x)
        # print(out.shape)
        out = self.trunk_output(out)
        # print(out.shape)
        # # print("x shape: {}".format(x.shape))
        # x = self.first_conv(x)
        # x_s4 = self.layer2_3(x)
        # # print("x_s4 shape: {}".format(x_s4.shape))
        # x_s8 = self.layer4(x_s4)
        # # print("x_s8 shape: {}".format(x_s8.shape))
        # x_16 = self.layer5_6(x_s8)
        # # print("x_16 shape: {}".format(x_16.shape))

        # x = self.upconv3(x_16)
        # # print("x_s8 shape: {}".format(x_s8.shape))
        # # print("x shape: {}".format(x.shape))
        # x = torch.cat([x_s8, x], dim=1)
        # x = self.iconv3(x)
        # x = self.upconv2(x)
        # x = torch.cat([x_s4, x], dim=1)
        # x = self.iconv2(x)
        # x_fine = self.conv_fine(x)
        # return x_s8, x_fine
        # # return x_s8
        return out



def run():
    import time
    model = RegNetMagic()
    model.cuda()
    model.eval()

    run_time = 1000

    with torch.no_grad():
        in_size = [1, 3, 1080, 1080]
        dummy_input = torch.randn(*in_size, device='cuda')
        out = model(dummy_input)
        print("out.shape: {}".format(out.shape))

        torch.cuda.synchronize()
        start_time = time.time()
        for i in range(run_time):
            model(dummy_input)
        torch.cuda.synchronize()
        end_time = time.time() - start_time
    print("the time interval is :{}".format(end_time))


def m_test():
    model = RegNetMagic()
    model.cuda()
    model.eval()

    with torch.no_grad():
        in_size = [1, 3, 224, 224]
        dummy_input = torch.randn(*in_size, device='cuda')
        out = model(dummy_input)
        print("out.shape: {}".format(out.shape))


def to_onnx():
    efficient_unet = EfficientUNet()
    efficient_unet.cuda()
    efficient_unet.eval()
    in_name = ["data",]
    out_name = ["descriptor",]
    dummy_input = torch.randn([1, 3, 224, 224], device=torch.device("cuda:0"))
    onnx_model_path = "./efficient_unet.pth"
    torch.onnx.export(efficient_unet,
                      dummy_input,
                      onnx_model_path,
                      input_names=in_name,
                      output_names=out_name,
                      dynamic_axes=None,
                      export_params=True,
                      opset_version=11,
                      do_constant_folding=True,
                      verbose=True)


if __name__ == "__main__":
    run()
    # m_test()
    # to_onnx()