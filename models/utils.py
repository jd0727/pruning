import pynvml
import torch
import torchvision
import numpy as np
import sys
import torch.nn as nn

# 计算iou 输入为x1y1x2y2格式
def clac_iou_mat(boxes1, boxes2):
    num1 = boxes1.shape[0]
    num2 = boxes2.shape[0]
    # 面积
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area1 = np.expand_dims(area1, axis=1).repeat(num2, axis=1)
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    area2 = np.expand_dims(area2, axis=0).repeat(num1, axis=0)
    # 重叠w
    b1x2 = np.expand_dims(boxes1[:, 2], axis=1).repeat(num2, axis=1)
    b2x2 = np.expand_dims(boxes2[:, 2], axis=0).repeat(num1, axis=0)
    b1x1 = np.expand_dims(boxes1[:, 0], axis=1).repeat(num2, axis=1)
    b2x1 = np.expand_dims(boxes2[:, 0], axis=0).repeat(num1, axis=0)
    iw = np.minimum(b1x2, b2x2) - np.maximum(b1x1, b2x1)
    iw[iw < 0] = 0
    # 重叠h
    b1y2 = np.expand_dims(boxes1[:, 3], axis=1).repeat(num2, axis=1)
    b2y2 = np.expand_dims(boxes2[:, 3], axis=0).repeat(num1, axis=0)
    b1y1 = np.expand_dims(boxes1[:, 1], axis=1).repeat(num2, axis=1)
    b2y1 = np.expand_dims(boxes2[:, 1], axis=0).repeat(num1, axis=0)
    ih = np.minimum(b1y2, b2y2) - np.maximum(b1y1, b2y1)
    ih[ih < 0] = 0
    # 计算总面积
    areai = iw * ih
    areao = area1 + area2 - areai
    return areai / areao


#NMS
def nms(boxes,scores,iou_threshold=0.45):
    prserv_inds = torchvision.ops.nms(boxes, scores, iou_threshold=iou_threshold)
    return prserv_inds

#按类别NMS
def nms_by_cls(boxes,scores,pred_cls,iou_threshold=0.45,num_cls=80):
    prserv_inds = []
    for cls_ind in range(num_cls):
        inds=pred_cls==cls_ind
        if not torch.sum(inds)==0:
            boxes_cls=boxes[inds]
            scores_cls = scores[inds]
            prserv_inds_cls = torchvision.ops.nms(boxes_cls,scores_cls, iou_threshold=iou_threshold)
            #添加
            inds = torch.nonzero(inds,as_tuple=False).squeeze(dim=1)
            prserv_inds.append(inds[prserv_inds_cls])
    prserv_inds=torch.cat(prserv_inds,dim=0)
    return prserv_inds

#######################################################分配gpu占用
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
#小于min_thres的都会被占用
#上述不匹配，取最小一台，若该台占用小于one_thres
def ditri_gpu(min_thres=0.1,one_thres=0.3,show_version=False):
    #version
    if show_version:
        print("Python  version : {}".format(sys.version.replace('\n', ' ')))
        print("Torch   version : {}".format(torch.__version__))
        print("Vision  version : {}".format(torchvision.__version__))
        print("cuDNN   version : {}".format(torch.backends.cudnn.version()))
    #usage
    print("GPU usage")
    usages=get_cuda_usage()
    for i in range(len(usages)):
        percent=usages[i]*100
        print('    cuda:',str(i),' using:',"%3.3f" % percent,'%')
    #过滤
    if np.min(usages)<min_thres:
        inds = [i for i in range(len(usages)) if usages[i] < min_thres]
    elif np.min(usages)<one_thres:
        inds=[np.argmin(usages)]
    else:
        print('No available GPU')
        return None
    # 交换顺序device_ids[0]第一个出现
    inds=sorted(inds,key=lambda x:usages[x])
    return inds

def Focalloss(pred, target, alpha=0.5, gamma=2):
    prop = torch.where(target > 0.5, pred, 1 - pred)
    loss = -torch.pow((1 - prop), gamma) * torch.log(prop + 1e-7)
    loss = torch.where(target > 0.5, loss * alpha, loss)
    return torch.sum(loss)


def model2onnx(model, file_name, input_size):
    #规范输入
    if isinstance(input_size,int):
        test_input = torch.rand(size=(5, 3, input_size, input_size))*3
    elif len(input_size)==2:
        test_input = torch.rand(size=(5, 3, input_size[0], input_size[1]))*3
    elif len(input_size)==3:
        test_input = torch.rand(size=(5, input_size[0], input_size[1], input_size[2]))*3
    elif len(input_size)==4:
        test_input = torch.rand(size=input_size)*3
    else:
        print('err size')
        return 0
    if not str.endswith(file_name, '.onnx'):
        file_name += '.onnx'
    torch.onnx.export(model, test_input, file_name, verbose=True, opset_version=11)
    return None

####################################################################################################权重文件加载
def load_fmt(model,sd_ori='',character='fname',only_fullmatch=False):
    if isinstance(sd_ori,str):
        if hasattr(model,'device'):
            device=model.device
        else:
            device=torch.device('cpu')
        sd_ori = torch.load(sd_ori,map_location=device)
    sd_tar = model.state_dict()
    names_tar = list(sd_tar.keys())
    names_ori = list(sd_ori.keys())
    tensors_tar = list(sd_tar.values())
    tensors_ori = list(sd_ori.values())
    #匹配
    print('Try to match by', character)
    arr_tar = [[] for _ in range(len(names_tar))]
    arr_ori = [[] for _ in range(len(names_ori))]
    cert = lambda x, y: x == y
    fitted=False
    if 'size' in character:
        fitted=True
        for i in range(len(arr_tar)):
            arr_tar[i].append(tensors_tar[i].size())
        for i in range(len(arr_ori)):
            arr_ori[i].append(tensors_ori[i].size())
    if 'fname' in character:
        fitted = True
        for i in range(len(arr_tar)):
            arr_tar[i].append(names_tar[i])
        for i in range(len(arr_ori)):
            arr_ori[i].append(names_ori[i])
    if 'lname' in character:
        fitted = True
        for i in range(len(arr_tar)):
            arr_tar[i].append(names_tar[i].split('.')[-1])
        for i in range(len(arr_ori)):
            arr_ori[i].append(names_ori[i].split('.')[-1])
    if not fitted:
        print('Unknown character',character)
        return False
    fit_pairs = matcher(arr_tar, arr_ori, cert)
    #检查匹配结果
    print('* Total:', len(names_tar), '* Match:', len(fit_pairs))
    if only_fullmatch and len(fit_pairs)<len(names_tar):
        print('Not enough match')
        return False
    fit_sd = {}
    for i in range(len(fit_pairs)):
        i_tar, i_ori = fit_pairs[i]
        fit_sd[names_tar[i_tar]] = tensors_ori[i_ori]
    for name, tensor in sd_tar.items():
        if name not in fit_sd.keys():
            fit_sd[name] = tensor
    #匹配添加
    for name, tensor in fit_sd.items():
        names = str.split(name, '.')
        tar = model
        for n in names:
            tar = getattr(tar, n)
        tar.data=tensor
    return True

def matcher(arr1,arr2,cert=None):
    if cert is None:
        cert=lambda x,y:x==y
    num1,num2=len(arr1),len(arr2)
    fit_mat=np.full(shape=(num1,num2),fill_value=False)
    for i in range(num1):
        for j in range(num2):
            fit_mat[i,j]=cert(arr1[i],arr2[j])
    fit_pairs=[]
    for s in range(num1+num2):
        for i in range(s,-1,-1):
            j=s-i
            if i>=num1 or j>=num2 or j<0:
                continue
                #查找匹配
            if fit_mat[i,j]:
                fit_pairs.append([i,j])
                fit_mat[i,:]=False
                fit_mat[:,j]=False
                #print('Fit ',i,' --- ', j)
    return fit_pairs

def refine_chans(model):
    def refine(model):
        if len(list(model.children())) == 0:
            if isinstance(model,nn.Conv2d):
                wei=model.weight
                model.in_channels=wei.size()[1]
                model.out_channels=wei.size()[0]
            elif isinstance(model,nn.BatchNorm2d):
                wei = model.weight
                model.num_features=wei.size()[0]
                model.running_mean = model.running_mean[:wei.size()[0]]
                model.running_var = model.running_var[:wei.size()[0]]
            elif isinstance(model, nn.Linear):
                wei=model.weight
                model.in_features=wei.size()[1]
                model.out_features=wei.size()[0]
        else:
            for name, sub_model in model.named_children():
                refine(sub_model)

    refine(model)
    return None

if __name__ == '__main__':
    inds=ditri_gpu(min_thres=0.1,one_thres=0.5)