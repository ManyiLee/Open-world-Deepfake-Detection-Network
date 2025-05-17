import os
import torch
import sys
import logging
import numpy as np
import random 

def set_trainable(model, boolean: bool = True, except_layers: list = [], device_ids: list = []):
    if boolean:
        for name, param in model.named_parameters():
            if name in except_layers:
                param.requires_grad = False
            else:
                param.requires_grad = True
    else:
        #assert len(except_layers) > 0, "Require free layer"
        for name, param in model.named_parameters():
            if name in except_layers:
                param.requires_grad = True
                print('--------',name)
            else:
                param.requires_grad = False

    print("Training params: ", count_parameters(model))
    return model

def count_parameters(model):
    "Couting number of trainable params"
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def load_model(load_path:str, model, optimizer):
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    return epoch

def print_options(opt):
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(opt.items()):
        comment = ''
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)

    # save to the disk
    expr_dir = os.path.join(opt['checkpoints_dir'], opt['name'])
    mkdirs(expr_dir)
    file_name = os.path.join(expr_dir, 'opt.txt')
    with open(file_name, 'wt') as opt_file:
        opt_file.write(message)
        opt_file.write('\n')
        
def get_val_opt(val_opt:dict)->dict:

    """
    Currently assumes jpg_prob, blur_prob 0 or 1
    modify cfg to fit validation
    """
    val_opt['dataroot'] = '/root/limanyi/DeepfakeDetection_reimplement/CNNDetection/dataset/'
    val_opt['classes'] = ['airplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable',
                    'dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']

    val_opt['dataroot'] = os.path.join(val_opt['dataroot'], val_opt['val_split']) 
    val_opt['isTrain'] = False
    val_opt['no_resize'] = False
    val_opt['no_crop'] = True
    val_opt['serial_batches'] = True
    
    val_opt['blur_prob'] = 1.0
    val_opt['mode'] = "RandomCmp"
    #val_opt['jpg_method'] = ['cv2']
    #val_opt['jpg_prob'] = 1.0
    #val_opt['jpg_qual'] = range(30,100)
    return val_opt

def seed_everything(seed):
    """
    constrain random seed and promise same setting with same result
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
# a function  to create and save logs in the log files
def log(path, file):
    """[Create a log file to record the experiment's logs]
    
    Arguments:
        path {string} -- path to the directory
        file {string} -- file name
    
    Returns:
        [obj] -- [logger that record logs]
    """

    # check if the file exist
    log_file = os.path.join(path, file)

    if not os.path.isfile(log_file):
        open(log_file, "w+").close()

    # console_logging_format = "%(levelname)s %(message)s"
    # file_logging_format = "%(levelname)s: %(asctime)s: %(message)s"
    console_logging_format = "%(message)s"
    file_logging_format = "%(message)s"
    # configure logger
    logging.basicConfig(level=logging.INFO, format=console_logging_format)
    logger = logging.getLogger()
    
    # create a file handler for output file
    handler = logging.FileHandler(log_file)

    # set the logging level for log file
    handler.setLevel(logging.INFO)
    
    # create a logging format
    formatter = logging.Formatter(file_logging_format)
    handler.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(handler)

    return logger
    
def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def unnormalize(tens, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    # assume tensor of shape NxCxHxW
    return tens * torch.Tensor(std)[None, :, None, None] + torch.Tensor(
        mean)[None, :, None, None]


class Logger(object):
    """Log stdout messages."""

    def __init__(self, outfile):
        self.terminal = sys.stdout
        self.log = open(outfile, "a")
        sys.stdout = self

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()

