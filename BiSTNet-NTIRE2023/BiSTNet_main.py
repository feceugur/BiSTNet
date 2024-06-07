import argparse
import os
import numpy as np
import torch
from torch.backends import cudnn
from PIL import Image
from BiSTNet_functions import WarpNet_debug
from BiSTNet_functions import ColorVidNet
from BiSTNet_functions import VGG19_pytorch
from BiSTNet_functions import ColorVidNet_wBasicVSR_v3
from BiSTNet_functions import RAFT
from BiSTNet_functions import ATB
from BiSTNet_functions import load_pth
from BiSTNet_functions import Hed
from BiSTNet_functions import superslomo_transforms
from BiSTNet_functions import exists_or_mkdir
from BiSTNet_functions import colorize_video




flag_ntire23 = True  # else use DAVIS raw ref dataset structure
flag_ntire23_OOMSplitVideo = False  # else use DAVIS raw ref dataset structure && split videos to F300 F300-600 F600
flag_ntire23_OOMSplitVideo_v2Automatic = True  # split videos to len_interval

epoch = 105000
dirName_ckp = '20230311_NTIRE2023'
nonlocal_test_path = os.path.join("checkpoints/", "finetune_test0610/nonlocal_net_iter_6000.pth")
color_test_path = os.path.join("checkpoints/", "finetune_test0610/colornet_iter_6000.pth")
fusenet_path = os.path.join("checkpoints/", "%s/fusenet_iter_%s.pth" % (dirName_ckp, epoch))
atb_path = os.path.join("checkpoints/", "%s/atb_iter_%s.pth" % (dirName_ckp, epoch))

parser = argparse.ArgumentParser()

# Add the input_data argument
parser.add_argument("--input_data", type=str, required=True, help="Path of input clips")

# Parse the arguments
args = parser.parse_args()

# Access the input_data argument
input_data = args.input_data

# Create the paths
input_dataset = f"../{input_data}/input/001/"
ref_dataset = f"../{input_data}/ref/001/"
out_dir = f"../{input_data}/output/001/"

parser.add_argument(
    "--frame_propagate", default=False, type=bool, help="propagation mode, , please check the paper"
)
parser.add_argument("--image_size", type=int, default=[448, 896], help="the image size, eg. [216,384]")
parser.add_argument("--cuda", action="store_false")
parser.add_argument("--gpu_ids", type=str, default="0", help="separate by comma")

# 20230215 ntire test set
parser.add_argument("--clip_path", type=str, default=input_dataset, help="path of input clips")
parser.add_argument("--ref_path", type=str, default=ref_dataset, help="path of refernce images")
parser.add_argument("--output_path", type=str, default=out_dir, help="path of output clips")

start_idx = 0
end_idx = -1

# RAFT params
parser.add_argument('--model', default='data/raft-sintel.pth', type=str, help="restore checkpoint")
parser.add_argument('--path', help="dataset for evaluation")
parser.add_argument('--small', action='store_true', help='use small model')
parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')

opt = parser.parse_args()
opt.gpu_ids = [int(x) for x in opt.gpu_ids.split(",")]
cudnn.benchmark = True
print("running on GPU", opt.gpu_ids)

opt_clip_path = opt.clip_path
opt_ref_path = opt.ref_path
opt_output_path = opt.output_path

nonlocal_net = WarpNet_debug(1)
colornet = ColorVidNet(7)
vggnet = VGG19_pytorch()
fusenet = ColorVidNet_wBasicVSR_v3(33, flag_propagation=False)

### Flownet: raft version
flownet = RAFT(opt)

### ATB
atb = ATB()

vggnet.load_state_dict(torch.load("data/vgg19_conv.pth"))
for param in vggnet.parameters():
    param.requires_grad = False

load_pth(nonlocal_net, nonlocal_test_path)
load_pth(colornet, color_test_path)
load_pth(fusenet, fusenet_path)
load_pth(flownet, opt.model)
load_pth(atb, atb_path)
print("succesfully load nonlocal model: ", nonlocal_test_path)
print("succesfully load color model: ", color_test_path)
print("succesfully load fusenet model: ", fusenet_path)
print("succesfully load flownet model: ", 'raft')
print("succesfully load atb model: ", atb_path)

fusenet.eval()
fusenet.cuda()
flownet.eval()
flownet.cuda()
atb.eval()
atb.cuda()
nonlocal_net.eval()
colornet.eval()
vggnet.eval()
nonlocal_net.cuda()
colornet.cuda()
vggnet.cuda()

opt_image_size = opt.image_size

# HED
hed = Hed().cuda().eval()
w0, h0 = opt_image_size[0], opt_image_size[1]
w, h = (w0 // 32) * 32, (h0 // 32) * 32
# forward l
intWidth = 480
intHeight = 320
meanlab = [-50, -50, -50]  # (A - mean) / std
stdlab = [100, 100, 100]  # (A - mean) / std
trans_forward_hed_lll = superslomo_transforms.Compose(
    [superslomo_transforms.Normalize(mean=meanlab, std=stdlab), superslomo_transforms.Resize([intHeight, intWidth])])
# backward
trans_backward = superslomo_transforms.Compose([superslomo_transforms.Resize([w0, h0])])

# proto seg
meanlab_protoseg = [0.485, 0.485, 0.485]  # (A - mean) / std
stdlab_protoseg = [0.229, 0.229, 0.229]  # (A - mean) / std
trans_forward_protoseg_lll = superslomo_transforms.Compose([superslomo_transforms.Normalize(mean=meanlab, std=stdlab),
                                                            superslomo_transforms.Normalize(mean=meanlab_protoseg,
                                                                                            std=stdlab_protoseg)])

# dataset preprocessing for batch testing
clips = sorted(os.listdir(opt_clip_path))
opt_clip_path_ori = opt_clip_path
opt_ref_path_ori = opt_ref_path
opt_output_path_ori = opt_output_path

for idx_clip, clip in enumerate(clips):
    dirTestImageName = os.path.join(opt_clip_path_ori, sorted(os.listdir(opt_clip_path_ori))[idx_clip])
    TestImageName = os.path.join(opt_clip_path_ori, sorted(os.listdir(opt_clip_path_ori))[idx_clip],
                                 os.listdir(dirTestImageName)[0])
    test_img = Image.open(TestImageName).convert('RGB')
    opt_image_size_ori = np.shape(test_img)[:2]

    opt_image_size = opt.image_size

    dirName_input = os.path.join(opt_clip_path_ori, clip)
    dirName_ref = os.path.join(opt_ref_path_ori, clip)
    dirName_output = os.path.join(opt_output_path_ori, clip)

    opt_clip_path = dirName_input
    opt_ref_path = dirName_ref
    opt_output_path = dirName_output

    print(idx_clip, clip, opt_clip_path, opt_ref_path, opt_output_path)

    exists_or_mkdir(dirName_output)
    clip_name = opt_clip_path.split("/")[-1]
    refs = os.listdir(opt_ref_path)
    refs.sort()

    ref_name = refs[start_idx].split('.')[0] + '_' + refs[end_idx].split('.')[0]

    len_interval = 50
    flag_lf_split_test_set = True

    for i in range(0, len(refs), len_interval):
        if i != 0:
            sub_ref = refs[i - 1:i + len_interval]
            ActStartIdx = i - 1
            ActEndIdx = i + len_interval
        else:
            sub_ref = refs[i:i + len_interval]
            ActStartIdx = i
            ActEndIdx = i + len_interval
        ActEndIdx = min(ActEndIdx, len(refs))

        print(i, 'startImg: %s endImg: %s, ActStartIdx: %s, ActEndIdx: %s' % (
            sub_ref[0], sub_ref[-1], ActStartIdx, ActEndIdx))

        colorize_video(
            opt,
            opt_clip_path,
            [os.path.join(opt_ref_path, name) for name in refs],
            # os.path.join(opt_output_path, clip_name + "_" + ref_name.split(".")[0]),
            os.path.join(opt_output_path),
            nonlocal_net,
            colornet,
            fusenet,
            vggnet,
            flownet,
            flag_lf_split_test_set,
            ActStartIdx,
            ActEndIdx,
        )
