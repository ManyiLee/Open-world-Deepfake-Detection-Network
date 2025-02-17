import os
import cv2
import numpy as np
import os
from PIL import Image
import sys
import random
import argparse
from tqdm import tqdm

def lower_quality(img, compress_val):
    img = np.array(img)
    img_cv2 = img[:,:,::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode('.jpg', img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return Image.fromarray(decimg[:,:,::-1])

def init_9Gans(mode:str, ratio:float, down:int, up:int):
    vals = ['AttGAN', 'BEGAN', 'CramerGAN', 'InfoMaxGAN', 'MMDGAN', 'RelGAN', 'S3GAN', 'SNGAN', 'STGAN']
    multiclass = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    
    for v_id, val in enumerate(vals):
        dataroot = '/opt/data/private/limanyi/DeepfakeDetection_reimplement/CNNDetection/dataset/test'
        dataroot = '{}/{}/NoCmp'.format(dataroot, val)
        classes = os.listdir(dataroot) if multiclass[v_id] else ['']
        image_list = []
        for cls in classes:
            root = os.path.join(dataroot, cls)
            print(root)
            for root, dirs, files in os.walk(root): 
                for file in files:  
                    file_path = os.path.join(root, file)
                    image_list.append(file_path)
        
        sample_num = int(len(image_list)*ratio)
        sample_list = random.sample(image_list, sample_num)
        for file_path in tqdm(sample_list):
            save_path = file_path.replace("NoCmp", mode)
            image = Image.open(file_path)
            os.makedirs(os.path.dirname(save_path),exist_ok=True)
            if image.mode == 'L':  
                image = image.convert('RGB')
                
            image = lower_quality(image, random.randint(down, up))
            image.save(save_path)

def init_8Gans(mode:str, ratio:float, down:int, up:int):
    vals = ['progan', 'stylegan', 'stylegan2', 'biggan', 'cyclegan', 'stargan', 'gaugan', 'deepfake']
    multiclass = [1, 1, 1, 0, 1, 0, 0, 0]
    
    for v_id, val in enumerate(vals):
        dataroot = '/opt/data/private/limanyi/DeepfakeDetection_reimplement/CNNDetection/dataset/test'
        dataroot = '{}/{}/NoCmp'.format(dataroot, val)
        classes = os.listdir(dataroot) if multiclass[v_id] else ['']
        image_list = []
        for cls in classes:
            root = os.path.join(dataroot, cls)
            print(root)
            for root, dirs, files in os.walk(root): 
                for file in files:  
                    file_path = os.path.join(root, file)
                    image_list.append(file_path)
        
        sample_num = int(len(image_list)*ratio)
        sample_list = random.sample(image_list, sample_num)
        for file_path in tqdm(sample_list):
            save_path = file_path.replace("NoCmp", mode)
            image = Image.open(file_path)
            os.makedirs(os.path.dirname(save_path),exist_ok=True)
            if image.mode == 'L':  
                image = image.convert('RGB')
                
            image = lower_quality(image, random.randint(down, up))
            image.save(save_path)

def init_ProGan(mode:str, data_type:str, ratio:float, down:int, up:int):
    image_list = []
    
    dataroot = '/opt/data/private/limanyi/DeepfakeDetection_reimplement/CNNDetection/dataset/{}/NoCmp'.format(data_type)
    classes = os.listdir(dataroot)
    for cls in classes:
        root = os.path.join(dataroot, cls)
        print(root)
        for root, dirs, files in os.walk(root): 
            for file in files:  
                file_path = os.path.join(root, file)
                image_list.append(file_path)
        
    sample_num = int(len(image_list)*ratio)
    sample_list = random.sample(image_list, sample_num)
    for file_path in tqdm(sample_list):
        save_path = file_path.replace("NoCmp", mode)
        image = Image.open(file_path)
        os.makedirs(os.path.dirname(save_path),exist_ok=True)
        if image.mode == 'L':  
            image = image.convert('RGB')
                
        image = lower_quality(image, random.randint(down, up))
        image.save(save_path)

def main(args):
    if args.dataset == '9Gans':
        init_9Gans(args.mode, args.ratio, args.down, args.up)
    elif args.dataset == '8Gans':
        init_8Gans(args.mode, args.ratio, args.down, args.up)
    elif args.dataset == 'ProGan':
        init_ProGan(args.mode, args.type, args.ratio, args.down, args.up)
    else:
        NotImplementedError






if __name__=='__main__':


    parser=argparse.ArgumentParser()
    parser.add_argument('-r',dest='ratio',type=float,default=1.0)
    parser.add_argument('-d',dest='dataset',type=str,default='9Gans')
    parser.add_argument('-m',dest='mode',type=str,default='RandomCmp')
    parser.add_argument('-t',dest='type',type=str,default='train')
    parser.add_argument('-up',dest='up',default=100,type=int)
    parser.add_argument('-down',dest='down',default=30,type=int)
    args=parser.parse_args()

    seed=1
    random.seed(seed)
    np.random.seed(seed)

    main(args)