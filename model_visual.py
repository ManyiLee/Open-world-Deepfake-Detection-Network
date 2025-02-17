import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
device_ids=[0, 1]
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, transforms
from tqdm import tqdm  

from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.pyplot as plt 
from networks.ODDN import ODDN
from options.train_options import TrainOptions

def cv2_jpg(img, compress_val):
    img_cv2 = img[:,:,::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode('.jpg', img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:,:,::-1]

def lower_quality(img, compress_val):
    img = np.array(img)
    img = cv2_jpg(img,compress_val)
    return Image.fromarray(img)


def visualize():
    opt = TrainOptions().parse()

    #baseline:/opt/data/private/limanyi/AAAI2024/iccv+ours/checkpoints/ablation-baseline-30n2024_07_28_05_32_22/model_epoch_28.pth
    state_dict = torch.load("/opt/data/private/limanyi/AAAI2024/iccv+ours/checkpoints/ablation-baseline-30n2024_07_28_05_32_22/model_epoch_28.pth", map_location='cpu')
    model = ODDN(opt)
    model.load_state_dict(state_dict["model"])
    model = model.to('cuda')

    for name, _ in model.named_parameters():
        print(name, _.shape)
    #  model.resnet.module.layer4.2.bn3
    target_layers = [model.resnet.module.layer4[-1].bn3]         #定义目标层
 
    data_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    # load image        
    img_path = "/opt/data/private/limanyi/AAAI2024/iccv+ours/visual/images/fake/000014.png"  
    
    img = Image.open(img_path).convert('RGB')
    img = img.resize((224,224))
    img = lower_quality(img, 60)
    img = np.array(img, dtype=np.uint8)
    img_tensor = data_transform(img)

    input_tensor = torch.unsqueeze(img_tensor, dim=0).repeat(1,1,1,1)
 
    #cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)
    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=input_tensor)
 
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255., grayscale_cam, use_rgb=True)
    plt.imshow(visualization)
    plt.savefig("./visual/{}-baseline-cmp60.png".format(img_path.split('/')[-2:]), dpi=300)


if __name__ == "__main__":

    visualize()