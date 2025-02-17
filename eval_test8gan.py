import time
import os
import yaml
import torch
import numpy as np
import warnings

from util import Logger
from validate import validate
from networks.ODDN import ODDN
from util import get_val_opt
from copy import deepcopy
warnings.filterwarnings("ignore")

vals = ['progan', 'stylegan', 'stylegan2', 'biggan', 'cyclegan', 'stargan', 'gaugan', 'deepfake']
multiclass = [1, 1, 1, 0, 1, 0, 0, 0]

def test_8GANs(model, opt, cmp = False):

    opt['dataroot'] = '/opt/data/private/limanyi/DeepfakeDetection_reimplement/CNNDetection/dataset/test'
    dataroot = opt['dataroot']
    opt['batch_size'] = 32
    if cmp:
        if opt['agnostic']:
            opt['mode'] = 'RandomCmp'
        else:
            opt['mode'] = 'StaticCmp'
    else: 
        opt['mode'] = 'NoCmp' 
    
    accs = {}; aps = {}
    model.eval()
    with torch.no_grad():
        for v_id, val in enumerate(vals):
            opt['dataroot'] = '{}/{}/{}'.format(dataroot, val, opt['mode'])
            opt['classes'] = os.listdir(opt['dataroot']) if multiclass[v_id] else ['']
            opt['no_resize'] = False    # testing without resizing by default
            opt['no_crop'] = True    # testing without resizing by default

            acc, ap, auc, _, _, _, _, _ = validate(model, opt)
            accs[val] = acc * 100; aps[val] = ap * 100
    
    avg_acc , avg_ap = [ value  for value in accs.values()],  [ value  for value in aps.values()]
    avg_acc , avg_ap = sum(avg_acc) / len(avg_acc), sum(avg_ap) / len(avg_ap)

    return accs, aps, avg_acc, avg_ap


if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"
    
    model_path = '/opt/data/private/limanyi/journal2025/ODDN/checkpoints/NoCmp-introduction/model_10_9+8Gan_72.60.pth'

    results_dir = './results_onprogan/'
    logpath = os.path.join(results_dir, model_path.split('/')[-2])
    os.makedirs(results_dir, mode = 0o777, exist_ok = True) 
    os.makedirs(logpath, mode = 0o777, exist_ok = True) 
    Logger(os.path.join(logpath, model_path.split('/')[-1] + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())+'.log'))

    dataroot = '/opt/data/private/limanyi/DeepfakeDetection_reimplement/CNNDetection/dataset/test'
    print(f'Dataroot {dataroot}')
    print(f'Model_path {model_path}')

    accs = [];aps = [];aucs=[]
    print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))

    with open('./configs/run.yaml', 'r') as file:  
        train_cfg = yaml.safe_load(file)
    val_cfg = deepcopy(train_cfg)
    val_cfg = get_val_opt(val_cfg)
    test_cfg = deepcopy(val_cfg)
    
    model = ODDN(train_cfg).to('cuda')
    
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict['model'])
    model.to('cuda')
    model.eval()

    accs, aps, avg_acc, avg_ap = test_8GANs(model, test_cfg, True)

    for v_id, val in enumerate(vals):
        print("({} {:10}) acc: {:.2f}; ap: {:.2f}".format(v_id, val, accs[val], aps[val]))
    print("({} {:10}) acc: {:.2f}; ap: {:.2f}".format(v_id+1,'Mean', avg_acc, avg_ap))
    print('*'*25)
