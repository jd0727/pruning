import torch
import numpy as np
import torch.nn as nn

#######################################################通道操作函数
#各个层的操作方式
def opr_conv_out(conv, opr_type,opr_args):
    conv.weight.data=opr_dim(conv.weight.data, opr_type,opr_args,dim=0)
    conv.out_channels = conv.weight.shape[0]
    if conv.bias is not None:
        conv.bias.data = opr_dim(conv.bias.data, opr_type, opr_args, dim=0)
    return None
def opr_conv_in(conv,opr_type, opr_args):
    conv.weight.data=opr_dim(conv.weight.data, opr_type,opr_args,dim=1)
    conv.in_channels = conv.weight.shape[1]
    return None
def opr_lin_out(lin,opr_type, opr_args):
    lin.weight.data=opr_dim(lin.weight.data,opr_type, opr_args,dim=0)
    if lin.bias is not None:
        lin.bias.data = opr_dim(lin.bias.data, opr_type, opr_args, dim=0)
    lin.out_features = lin.weight.shape[0]
    return None
def opr_lin_in(lin, opr_type,opr_args):
    lin.weight.data=opr_dim(lin.weight.data,opr_type, opr_args,dim=1)
    lin.in_features = lin.weight.shape[1]
    return None
def opr_bn(bn,opr_type, opr_args):
    for attr_name in ['running_mean','running_var','weight','bias']:
        attr=getattr(bn,attr_name)
        setattr(attr,'data',opr_dim(attr.data, opr_type,opr_args,dim=0))
    bn.num_features = bn.weight.shape[0]
    return
#操作对应关系
opr_dict={
    'in':{
        nn.Conv2d:opr_conv_in,
        nn.Linear: opr_lin_in,
    },
    'out':{
        nn.Conv2d:opr_conv_out,
        nn.Linear: opr_lin_out,
    },
    'pth':{
        nn.BatchNorm2d: opr_bn,
    }
    }
#张量维度操作
def opr_dim(data, opr_type,opr_args, dim=0):
    if not dim==0:
        data=data.transpose(dim, 0)
    size = data.shape[0]
    #操作类型
    if opr_type == 'ext':
        inds=opr_args['inds']
        data = data[inds]
    elif opr_type == 'frz':
        inds = opr_args['inds']
        data[inds].require_grad = False
    elif opr_type == 'rmv':
        inds = opr_args['inds']
        inds_inv = inv_inds(inds, size)
        data = data[inds_inv]
    elif opr_type == 'frmv':
        inds = opr_args['inds']
        data[inds] = 0
        data[inds].require_grad = False
    elif opr_type == 'zero':
        inds = opr_args['inds']
        data[inds] = 0
    elif opr_type == 'mul':
        inds = opr_args['inds']
        pow=opr_args['pow']
        data[inds] *= pow
    elif opr_type == 'cmb':#组合
        inds1 = opr_args['inds1']
        inds2 = opr_args['inds2']
        inds1_inv=inv_inds(inds1, size)
        data[inds2]+=data[inds1]
        #对于输出，要取均值
        if dim==0:
            data[inds2]/=2
        #rmv
        data = data[inds1_inv]
    elif opr_type == 'fcmb':#组合
        inds1 = opr_args['inds1']
        inds2 = opr_args['inds2']
        data[inds2]+=data[inds1]
        #对于输出，要取均值
        if dim==0:
            data[inds2]/=2
        data[inds1]=0

    elif opr_type == 'excg':#交换
        inds1 = opr_args['inds1']
        inds2 = opr_args['inds2']
        if dim==1:#对于接收通道
            tmp=data[inds1]
            data[inds1]=data[inds2]
            data[inds2]=tmp
    elif opr_type == 'linc':#线性组合替代
        rmv_ind=opr_args['rmv_ind']
        rmv_pow = opr_args['rmv_pow']
        inds_presv = inv_inds([rmv_ind], size)
        if dim==0:#对于输出通道直接剔除
            data = data[inds_presv]
        elif dim==1:#对于输入通道组合变换后剔除
            detla=data[rmv_ind:rmv_ind+1].repeat(size,1,1,1)
            for i in range(size):
                detla[i]*=rmv_pow[i]
            data += detla
            data = data[inds_presv]
    else:
        raise Exception('err type ')
    #交换还原
    if not dim==0:
        data=data.transpose(dim, 0)
    return data
#反选
def inv_inds(inds,size):
    inds_inv=[i for i in range(size) if not i in inds]
    return inds_inv
#通道操作
def chan_opr(binding_dict, opr_type,**opr_args):
    if opr_type=='fcmbp':
        inds1 = opr_args['inds1']
        inds2 = opr_args['inds2']
        joint_chan_opr(binding_dict,inds1,inds2)
        return None
    # 进行操作
    for edge in ['in','out','pth']:
        if edge in binding_dict.keys():
            models=binding_dict[edge]
            for model in models:
                opr_func=opr_dict[edge][type(model)]
                opr_func(model,opr_type,opr_args)
    return None

#联合操作
def joint_chan_opr(binding_dict,inds1,inds2):
    convs=binding_dict['out']
    assert len(convs)==1,'len err'

    conv = convs[0]

    #输出
    conv_w=conv.weight.data
    for ind1, ind2 in zip(inds1, inds2):
        v1 = conv_w[ind1].detach().reshape(-1)
        v2 = conv_w[ind2].detach().reshape(-1)
        #
        p1 = torch.sqrt(torch.mean(v1 ** 2))
        p2 = torch.sqrt(torch.mean(v2 ** 2))
        if p1 ==0 or p2==0:
            continue
        #根据p大小交换索引 p2>p1
        if p1>p2:
            p1,p2=(p2,p1)
            ind1,ind2=(ind2,ind1)
        #合并通道
        conv_w[ind2] = (conv_w[ind1] * p1 + conv_w[ind2] * p2) / (p1 + p2)
        conv_w[ind1] = 0

        if 'pth' in binding_dict.keys():
            bns = binding_dict['pth']
            assert len(bns) == 1, 'len err'
            bn = bns[0]
            #合并BN通道
            for attr_name in ['running_mean', 'running_var', 'weight', 'bias']:
                attr = getattr(bn, attr_name)
                attr.data[ind2]=(attr.data[ind1] * p1 + attr.data[ind2] * p2) / (p1 + p2)
                attr.data[ind1]=0
        #合并输出通道
        if binding_dict['type']=='inner':
            rec=binding_dict['in'][0]
            rec_w=rec.weight.data
            rec_w[: , ind2]=(rec_w[:,ind1] * p1 + rec_w[:,ind2] * p2) / (p1 + p2)
            rec_w[:, ind1]=0
    return None


#######################################################
#去除特定通道
def rmv_chans(binding_dicts, imp_list,opr_type='rmv'):
    #合并
    opr_indss= {}
    for chan_dict in imp_list:
        dict_ind=chan_dict['dict_ind']
        chan_ind = chan_dict['chan_ind']
        if not dict_ind in opr_indss.keys():
            opr_indss[dict_ind]=[chan_ind]
        else:
            opr_indss[dict_ind].append(chan_ind)
    #剪枝
    for dict_ind,opr_inds in opr_indss.items():
        chan_opr(binding_dicts[dict_ind],inds=opr_inds,opr_type=opr_type)
    return None

#合并通道
def cmb_chans(binding_dicts,sim_pairs,opr_type='cmb'):
    # 合并
    opr_indss1 = {}
    opr_indss2 = {}
    for sim_pair in sim_pairs:
        dict_ind = sim_pair['dict_ind']
        chan_inds = sim_pair['chan_inds']
        if not dict_ind in opr_indss1.keys():
            opr_indss1[dict_ind] = [chan_inds[0]]
            opr_indss2[dict_ind] = [chan_inds[1]]
        else:
            opr_indss1[dict_ind].append(chan_inds[0])
            opr_indss2[dict_ind].append(chan_inds[1])
    # 剪枝
    for dict_ind in opr_indss1.keys():
        chan_opr(binding_dicts[dict_ind], inds1=opr_indss1[dict_ind],
                 inds2=opr_indss2[dict_ind], opr_type=opr_type)
    return None

#######################################################剪枝确定函数

#全选择
def all_choice(binding_dicts):
    presv_indss=[]
    for i, binding_dict in enumerate(binding_dicts):
        size=binding_dict['size']
        inds=np.arange(0,size)
        presv_indss.append(inds)
    return presv_indss

#随机选择
def rand_choice(binding_dicts,presv_ratio=0.5):
    presv_indss=[]
    for i, binding_dict in enumerate(binding_dicts):
        size=binding_dict['size']
        inds=np.random.choice(size,int(size*presv_ratio),replace=False)
        presv_indss.append(inds)
    return presv_indss






#######################################################模型整体操作
#清除hook
def clear_hook(model):
    for sub_model in model.children():
        if len(list(sub_model.children())) == 0:
            sub_model._forward_pre_hooks.clear()
            sub_model._backward_hooks.clear()
            sub_model._forward_pre_hooks.clear()
        elif isinstance(sub_model, torch.nn.Module):
            clear_hook(sub_model)
        return

#重构模型去除不必要的层
def refmt_res(res_model):
    #得到stage数量
    stage_num = 4 if hasattr(res_model, 'stage4') else 3
    for s in range(1,stage_num+1):
        stage=getattr(res_model,'stage'+str(s))
        rmv_inds=[]
        for i, block in enumerate(stage.children()):
            if block.conv1.out_channels==0:
                rmv_inds.append(i)
            if hasattr(block,'conv3'):
                if block.conv2.out_channels==0:
                    rmv_inds.append(i)
        # 执行删除
        rmv_inds = rmv_inds[::-1]
        for i in rmv_inds:
            print('rmv block', i, 'in stage', s)
            stage.__delitem__(i)
    return None



#######################################################对通道进行评价
#计算熵
def calc_enp(ave_feat_map):
    #确保数值大于0
    if np.min(ave_feat_map)<0:
        ave_feat_map-=np.min(ave_feat_map)
    ######################归一化，再求std，再求均值
    ave_feat_map = ave_feat_map.reshape(ave_feat_map.shape[0], ave_feat_map.shape[1], -1)
    ave_feat_map=np.average(ave_feat_map,axis=2)
    std_val=np.std(ave_feat_map,axis=0)
    return std_val

def calc_enp2(ave_feat_map):
    #确保数值大于0
    if np.min(ave_feat_map)<0:
        ave_feat_map-=np.min(ave_feat_map)
    ######################归一化，再求std，再求均值
    ave_feat_map = ave_feat_map.reshape(ave_feat_map.shape[0], ave_feat_map.shape[1], -1)
    ave_feat_map = np.std(ave_feat_map, axis=0)
    ave_feat_map=np.average(ave_feat_map,axis=1)
    return ave_feat_map

#######################################################依据重要性选择通道


#从imp_list按序提取移除通道
def rmv_list_by_order(imp_list, num=5):
    imps = [imp_dict['imp'] for imp_dict in imp_list]
    full_order = np.argsort(imps)
    imp_list_rmv = [imp_list[i] for i in full_order[:num]]
    return imp_list_rmv


#从imp_cmb按序提取保留通道
def ext_cmb_by_order(imp_cmb, ratio=0.5):
    prsev_dict={}
    for key in imp_cmb.keys():
        imps=imp_cmb[key]
        full_order = np.argsort(imps)
        prsev_num=int(ratio*len(imps))
        prsev_dict[key]=full_order[(len(imps)-prsev_num):]

    return prsev_dict

#######################################################

