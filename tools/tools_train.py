import torch
import torch.nn as nn
import logging
from logging.handlers import TimedRotatingFileHandler
import os
import sys
import numpy as np
import pynvml
import torchvision
import pandas as pd
##################################################################################模型检查
def check_grad(model,loader):
    model.eval()
    (imgs, batch_gts) = next(iter(loader))
    loss = model.get_loss(imgs, batch_gts)
    loss.backward()
    report=pd.DataFrame(columns=['Name','Grad'])
    for name,para in model.named_parameters():
        report=report.append({
            'Name':name,
            'Grad':para.grad.norm().item()
        },ignore_index=True)
    print(report)
    return report

def check_para(model):
    for para in model.parameters():
        if torch.any(torch.isnan(para)):
            print('nan occur in models')
            para.data = torch.where(torch.isnan(para), torch.full_like(para, 0.1), para)
        if torch.any(torch.isinf(para)):
            print('inf occur in models')
            para.data = torch.where(torch.isinf(para), torch.full_like(para, 0.1), para)
        max = torch.max(para).item()
        min = torch.min(para).item()
        print('range: ',min,' - ',max)
    return None
##################################################################################训练
#训练单例
def train_model_single(model, bx, by, num, optimizer):
    model.train()
    for n in range(num):
        # print('num ',n+1)
        loss = model(bx, by)
        #
        if torch.isnan(loss):
            print('nan occur in loss')
            continue
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=100)
        optimizer.step()
        if n%10==0:
            print(n, 'loss', loss.item())
    return None

########################################################logger

#记录缓存
def get_logger(log_file):
    if not os.path.exists(log_file):
        os.makedirs(log_file)

    logger = logging.getLogger(log_file)
    logger.setLevel(logging.INFO)

    sys_handler=logging.StreamHandler(sys.stdout)
    sys_handler.setLevel(logging.INFO)

    info_handler = TimedRotatingFileHandler(log_file, when='D',encoding='utf-8')
    info_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    info_handler.setFormatter(formatter)
    sys_handler.setFormatter(formatter)

    if len(logger.handlers)==0:
        logger.addHandler(info_handler)

    return logger

########################################################模型训练与测试

# 模型测试
def test_model(model, test_loader, nums=1):
    model.eval()
    if isinstance(nums, int):
        nums=[nums]
    total = 0
    correct =  np.zeros(shape=len(nums))
    for batch_x, batch_y in test_loader:
        pred = model(batch_x)
        batch_y = batch_y.to(model.device)
        order=torch.argsort(pred,dim=1,descending=True)
        for i,num in enumerate(nums):
            for j in range(order.size(0)):
                if batch_y[j] in order[j,:num]:
                    correct[i]+=1
        total += batch_y.size(0)
    acc=100.00 * correct / total
    if len(nums)==1:
        acc=acc[0]
    return acc

#得到平均精度
def get_aver_acc(msg,last_num=5):
    accs=[m['acc'] for m in msg[-last_num:]]
    accs=np.array(accs)
    acc=np.average(accs)
    std=np.std(accs)
    return acc,std

# 模型训练
def train_model(model, train_loader, test_loader, num_epoch=0, num_iter=0, acc=0, total_epoch=200,
                optimizer=None, scheduler=None, logger=None, file_name='',reg_loss=None,
                save_process=False):
    ###############简单输出类
    class samp_logger():
        def info(self,msg):
            print(msg)
    ###############输入规范化
    if optimizer is None:
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                    lr=0.1, momentum=0.9, weight_decay=5e-4)
    if scheduler is None:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150],
                                                         gamma=0.1,last_epoch=num_epoch)

    if logger is None:
        logger = samp_logger()
        # if file_name is '':
        #     logger=samp_logger()
        # else:
        #     logger=get_logger(log_file=file_name+'.log')

    if not reg_loss is None:
        reg_loss=reg_loss.cuda()

    if save_process and file_name != '':
        full_name = file_name + '_' + str(0)
        torch.save(model.state_dict(),full_name)
    ###############进行训练
    msg=[]
    logger.info('Start train')
    for n in range(total_epoch):
        num_epoch += 1
        logger.info('Epoch = %d' % num_epoch)
        lr_cur=optimizer.param_groups[0]['lr']
        logger.info('Learn rate = %.5f' % lr_cur)
        loss=None
        model.train()
        for i, (batch_x, batch_y) in enumerate(train_loader):

            loss = model.get_loss(batch_x, batch_y)
            loss = loss+reg_loss if not reg_loss is None else loss
            #
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
            optimizer.step()
            # 显示
            if i % 100 == 0:
                logger.info('Iter = %d' % num_iter+' Loss = %.5f' % loss)
            num_iter += 1
        # 评估
        logger.info('Start test')
        acc_r = test_model(model, test_loader)
        msg.append({'epoch':num_epoch,'acc':acc_r,'loss':float(loss),'lr':lr_cur})
        #
        logger.info('Accuracy = %.2f' % (acc_r))

        scheduler.step()
        # 保存
        if file_name == '':
            continue

        if  acc_r > acc:
            acc = acc_r
            full_name = file_name + '_best'
            torch.save(model.state_dict(),full_name)
            logger.info('save for best')
        # if num_epoch % 10 == 0 or num_epoch==total_epoch:
        #     full_name = file_name + '_chk'
        #     torch.save(model.state_dict(),full_name)
        #     if save_process:
        #         full_name = file_name + '_'+str(num_epoch)
        #         torch.save(model.state_dict(),full_name)
    return msg

#继续训练
def train_SGD_StepLR(model,train_loader,test_loader,total_epoch=10,num_epoch=0, num_iter=0, acc=0,
                     lr=0.1,momentum=0.9, weight_decay=5e-4,
                  milestones=None, gamma=0.1,file_name='',reg_loss=None, save_process=False):
    if milestones is None:
        milestones = [3,5]
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    msg=train_model(model,train_loader,test_loader,total_epoch=total_epoch,optimizer=optimizer,scheduler=scheduler,
                file_name=file_name,reg_loss=reg_loss,save_process=save_process,
                    num_epoch=num_epoch,num_iter=num_iter,acc=acc)
    return msg
##########################################分配GPU
#得到GPU占用
def get_cuda_usage():
    pynvml.nvmlInit()
    num_cuda=pynvml.nvmlDeviceGetCount()
    usages=[]
    for ind in range(num_cuda):
        handle = pynvml.nvmlDeviceGetHandleByIndex(ind)  # 0表示第一块显卡
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        usage = meminfo.used / meminfo.total
        usages.append(usage)
    return usages

#自动分配GPU
def ditri_gpu(thres=0.001):
    print("Python  version : {}".format(sys.version.replace('\n', ' ')))
    print("Torch   version : {}".format(torch.__version__))
    print("Vision  version : {}".format(torchvision.__version__))
    print("cuDNN   version : {}".format(torch.backends.cudnn.version()))
    print("GPU usage")
    usages=get_cuda_usage()
    for i in range(len(usages)):
        percent=usages[i]*100
        print('    cuda:',str(i),' using:',"%3.3f" % percent,'%')
    #过滤
    inds = [i for i in range(len(usages)) if usages[i] < thres]
    if len(inds)==0:
        print('No available GPU')
        return None
    inds=sorted(inds) #交换顺序device_ids[0]第一个出现
    return inds



##########################################
if __name__ == '__main__':
    pass

