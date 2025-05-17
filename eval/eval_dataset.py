import os
import torch

from validate import validate
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def test_mygen9GANs(model, opt, cmp = False):

    vals = ['AttGAN', 'BEGAN', 'CramerGAN', 'InfoMaxGAN', 'MMDGAN', 'RelGAN', 'S3GAN', 'SNGAN', 'STGAN']
    multiclass = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    opt['dataroot'] = '/root/limanyi/DeepfakeDetection_reimplement/CNNDetection/dataset/test'
    dataroot = opt['dataroot']
    opt['batch_size'] = 32
    opt['mode'] = opt['agnostic']
    
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

def test_8GANs(model, opt, cmp = False):

    vals = ['progan', 'stylegan', 'stylegan2', 'biggan', 'cyclegan', 'stargan', 'gaugan', 'deepfake']
    multiclass = [1, 1, 1, 0, 1, 0, 0, 0]
    
    opt['dataroot'] = '/root/limanyi/DeepfakeDetection_reimplement/CNNDetection/dataset/test'
    dataroot = opt['dataroot']
    opt['batch_size'] = 32
    opt['mode'] = opt['agnostic']
    
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


def test_5Diffusions(model, opt, cmp = False):

    vals = ['DALLE', 'guided-diffusion',  'improved-diffusion',  'midjourney', 'ddpm-google'] 
    multiclass = [0, 0, 0, 0, 1]

    opt['dataroot'] = '/root/limanyi/DeepfakeDetection_reimplement/CNNDetection/dataset/test'
    dataroot = opt['dataroot']
    opt['batch_size'] = 32
    opt['mode'] = opt['agnostic']
    
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

def test_8Diffusions(model, opt, cmp = False):

    vals = ['dalle', 'glide_100_10', 'glide_100_27', 'glide_50_27', 'guided', 'ldm_100', 'ldm_200', 'ldm_200_cfg']
    multiclass = [0, 0, 0, 0, 0, 0, 0, 0]

    opt['dataroot'] = '/root/limanyi/DeepfakeDetection_reimplement/CNNDetection/dataset/test'
    dataroot = opt['dataroot']
    opt['batch_size'] = 32
    opt['mode'] = opt['agnostic']
    
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