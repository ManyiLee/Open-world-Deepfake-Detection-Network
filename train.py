import os
import yaml
import warnings
import numpy as np
import argparse

from tqdm import tqdm
from copy import deepcopy

warnings.filterwarnings('ignore')
    
def initial():
    seed_everything(200371)
    # read yaml config file as dict
    with open('./configs/run.yaml', 'r') as file:  
        train_cfg = yaml.safe_load(file)  
    
    val_cfg = deepcopy(train_cfg)
    val_cfg = get_val_opt(val_cfg)
    test_cfg = deepcopy(val_cfg)

    train_cfg['dataroot'] = os.path.join(train_cfg['dataroot'], train_cfg['train_split'])
    data_loader = create_dataloader(train_cfg)
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)
    
    train_writer = SummaryWriter(os.path.join(train_cfg['checkpoints_dir'], train_cfg['name'], "train"))
    val_writer = SummaryWriter(os.path.join(train_cfg['checkpoints_dir'], train_cfg['name'], "val"))
    logger = log(path=os.path.join(train_cfg['checkpoints_dir'], train_cfg['name']), file="losses.logs")
    
    return train_cfg, val_cfg, test_cfg, data_loader, train_writer, val_writer, logger

def training_process():
    
    #============Config and Model initialization============#
    train_cfg, val_cfg, test_cfg, data_loader, train_writer, val_writer, logger= initial()
    print_options(train_cfg)
    
    model = ODDN(train_cfg).to('cuda')
    
    hard_share_param = ["backbone.module.fc.weight","backbone.module.fc.bias"]
    print("#----Warm up Parameter----#")
    for name, p in model.named_parameters():
        if "resnet" not in name:
            hard_share_param.append(name)
    set_trainable(model, False, hard_share_param, [])
    
    if len(train_cfg['device_ids']) > 1:
        model = torch.nn.DataParallel(model, device_ids=train_cfg['device_ids'])
        model = model.module

    if train_cfg['optim'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),lr=train_cfg['lr'], betas=(0.9, 0.999),eps=1e-08, weight_decay=1e-4,amsgrad=False)
    else:
        raise ValueError("optim should be [adam, sgd]")
    
    if len(train_cfg['device_ids']) > 1:
        optimizer = torch.nn.DataParallel(optimizer, device_ids=train_cfg['device_ids']).module
    
    if train_cfg['continue_train']:
        print("continue last training......\nload path：{}".format(train_cfg['load_path']))
        train_cfg['continue_epoch'] = load_model(train_cfg['load_path'], model, optimizer)
    
    pc_adam = PCGrad(train_cfg, model, optimizer, train_cfg['max_grad_norm'])
    
    early_stopping = EarlyStopping(patience=train_cfg['earlystop_epoch'], delta=0.0001, verbose=True)
    
    #===============Training================#
    avg_acc_best = 0
    for epoch in range(train_cfg['continue_epoch'] + 1, train_cfg['epoch']):
        model.train()
        
        train_loss, train_true, train_pred = 0, [], []
        
        if train_cfg['warming_up'] == epoch:
            set_trainable(model, True, [], train_cfg['device_ids'])
                
        for data in tqdm(data_loader):

            if epoch >= train_cfg['adv_warmup']:
                alpha = train_cfg['alpha']
            else:
                alpha = None
            
            model.total_steps += 1
            
            #0 ture or not compressed / 1 fake or compressed
            aug_input, no_aug_input, tf_label, cmp_label = data
            cmp_label_np = np.array(cmp_label)
            input = torch.cat([no_aug_input[cmp_label_np], aug_input[cmp_label_np], aug_input[~cmp_label_np]], dim=0)
            mask_label = np.ones(len(input)).astype(bool)
            mask_label[sum(cmp_label_np)*2:] = False
            tf_label_np = np.array(torch.cat([tf_label[cmp_label], tf_label[cmp_label], tf_label[~cmp_label]], dim=0)).astype(bool)
            label = [tf_label_np, mask_label]
            input = input.cuda()

            tf_loss, cmp_loss, dis_loss, tf_output = model(input, True, label, alpha)
            train_pred.extend(tf_output.sigmoid().flatten().tolist())
            train_true.extend(torch.tensor(tf_label_np).float().flatten().tolist())
            train_loss += tf_loss.item()
            
            loss = tf_loss
            
            if dis_loss is not None:
                loss += dis_loss
            pc_adam.zero_grad()            
            if epoch >= train_cfg['adv_warmup'] and cmp_loss is not None:
                pc_adam.set_loss(cmp_loss)
                pc_adam.pc_backward(epoch, [loss, cmp_loss])#gradient alignment
            else:
                pc_adam.pc_backward(epoch, [loss])
            pc_adam.step()
            
            if model.total_steps % train_cfg['loss_freq'] == 0:
                print("Train loss: {} dis_loss{}: cmp_loss: {},at step: {}".format(loss, dis_loss, cmp_loss ,model.total_steps))
                train_writer.add_scalar('loss', loss, model.total_steps)
        
        train_true, train_pred = np.array(train_true), np.array(train_pred)
        train_acc = accuracy_score(train_true, train_pred > 0.5)
        
        # Validation
        model.eval()
        acc, ap = validate(model, val_cfg)[:2]
        val_writer.add_scalar('accuracy', acc, model.total_steps)
        val_writer.add_scalar('ap', ap, model.total_steps)
        print("(Val @ epoch {}) acc: {}; ap: {}".format(epoch, acc, ap))

        log_text="Epoch {}/{} | train loss: {:.4f}, train acc: {:.4f}, val acc：{:.4f}, val ap：{:.4f}".format(
                        epoch+1,
                        train_cfg['epoch'],
                        train_loss/len(data_loader),
                        train_acc,
                        acc,
                        ap,
                        )

        save_filename = 'model_epoch_latest.pth' 
        state_dict = {'epoch':epoch, 'model': model.state_dict(),'optimizer': optimizer.state_dict()}
        save_path = os.path.join(os.path.join(train_cfg['checkpoints_dir'], train_cfg['name']), save_filename)
        torch.save(state_dict, save_path)
        
        all_acc_avg = 0
        #test9GANs
        if "9GANs" in train_cfg["eval_list"]:
            accs, aps, avg_acc, avg_ap = test_mygen9GANs(model, test_cfg, True)
            for key, value in accs.items():
                val_writer.add_scalar(key + "acc" + "cmp", value, epoch)
            for key, value in aps.items():
                val_writer.add_scalar(key + "ap" + "cmp", value, epoch)
            val_writer.add_scalar("avg_acc_9GANs" + "cmp", avg_acc, epoch)
            val_writer.add_scalar("avg_ap_9GANs" + "cmp", avg_ap, epoch)
            print("({} {:10}) acc: {:.2f}; ap: {:.2f}".format("test9GAN CMP",'Mean', avg_acc, avg_ap))
            log_text+="\n 9GANs acc: {:.4f}, 9GANs ap: {:.4f}".format(avg_acc, avg_ap)
            all_acc_avg += avg_acc
        
        #test8GANs
        if "8GANs" in train_cfg["eval_list"]:
            accs, aps, avg_acc, avg_ap = test_8GANs(model, test_cfg, True)
            for key, value in accs.items():
                val_writer.add_scalar(key + "acc" + "cmp", value, epoch)
            for key, value in aps.items():
                val_writer.add_scalar(key + "ap" + "cmp", value, epoch)
            val_writer.add_scalar("avg_acc_8GANs" + "cmp", avg_acc, epoch)
            val_writer.add_scalar("avg_ap_8GANs" + "cmp", avg_ap, epoch)
            print("({} {:10}) acc: {:.2f}; ap: {:.2f}".format("test8GAN CMP",'Mean', avg_acc, avg_ap))
            all_acc_avg += avg_acc
        
        all_acc_avg = all_acc_avg / len(train_cfg["eval_list"])
        log_text+="\n ALL AVG：{:.4f}".format(all_acc_avg)
        
        if all_acc_avg > avg_acc_best:
            avg_acc_best = all_acc_avg
            save_filename = 'model_{}_{:.2f}.pth'.format(epoch, avg_acc_best) 
            state_dict = {'epoch':epoch, 'model': model.state_dict(),'optimizer': optimizer.state_dict()}
            save_path = os.path.join(os.path.join(train_cfg['checkpoints_dir'], train_cfg['name']), save_filename)
            torch.save(state_dict, save_path)
        
        logger.info(log_text)
        early_stopping(acc, epoch, model, optimizer)
        if early_stopping.early_stop:
            cont_train = model.adjust_learning_rate(optimizer)
            if cont_train:
                print("Learning rate multiply 0.8 , continue training...")
                early_stopping = EarlyStopping(patience=train_cfg['earlystop_epoch'], delta=-0.0001, verbose=True)
            else:
                print("Early stopping.")
                break
            
if __name__ == '__main__':
    
    parser=argparse.ArgumentParser()
    parser.add_argument('-g',dest='gpu',default='2,3')
    args=parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu 
    
    import torch
    
    from tensorboardX import SummaryWriter
    from validate import validate
    from data import create_dataloader
    from earlystop import EarlyStopping
    from networks.ODDN import ODDN
    from eval.eval_dataset import *
    from sklearn.metrics import accuracy_score
    from util import *  
    from networks.pcgrad import PCGrad
    
    training_process()