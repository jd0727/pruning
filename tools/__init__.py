from .tools_plot import *
from .tools_train import *
from .channel_opr import *
from .tools_file import *
from .tools_ext import *
from .counter import *


#####################################################
#特征图+分类
def eval_imp_feat(model, binding_dicts, dict_ind=None,loader=None):
    batch_x, batch_y = next(iter(loader))
    feat_maps = get_feat_maps(model, binding_dicts, batch_x)
    if dict_ind is None:
        ave_maps = ave_by_cls(feat_maps, batch_y)
        enps = list_apply(ave_maps, calc_enp)
        return enps
    else:
        ave_maps = ave_by_cls(feat_maps[dict_ind:dict_ind+1], batch_y)
        enps = list_apply(ave_maps, calc_enp)
        return enps[0][0]

#L1norm
def eval_imp_l1norm(model,binding_dicts,dict_ind=None,loader=None):
    out_weis = get_wights(binding_dicts,wei_from='out')
    out_weis =list_apply(out_weis,lambda x:x.data.detach().cpu().numpy() if isinstance(x,torch.Tensor) else x)
    out_imp = list_apply(out_weis, lambda x: np.average(np.abs(x), axis=(1, 2, 3)))
    imp_cmb = imp_data2cmb(out_imp)
    if dict_ind is None:
        return imp_cmb
    else:
        return imp_cmb[dict_ind]

#L2norm
def eval_imp_l2norm(model,binding_dicts,dict_ind=None,loader=None):
    # if dict_ind is not None:
    #     binding_dicts=binding_dicts[dict_ind:dict_ind+1]
    out_weis = get_wights(binding_dicts,wei_from='out')
    out_imp = list_apply(out_weis, lambda x: np.average(np.square(x.reshape(x.shape[0],x.shape[1],-1)), axis=(1, 2)))
    imp_cmb = imp_data2cmb(out_imp)
    if dict_ind is None:
        return imp_cmb
    else:
        return imp_cmb[dict_ind]
#输入权重L1*特征图L1
def eval_imp_fmf(model, binding_dicts, dict_ind=None,loader=None):
    batch_x, batch_y = next(iter(loader))
    feat_maps = get_feat_maps(model, binding_dicts, batch_x,feat_from='in',rec='in')
    ave_maps=ave_by_batch(feat_maps)
    in_weis=get_wights(binding_dicts,wei_from='in')
    feat_imp = list_apply(ave_maps, lambda x: np.average(np.abs(x), axis=(1, 2)))
    in_imp = list_apply(in_weis, lambda x: np.average(np.abs(x), axis=(0, 2, 3)))
    mul_imp=list_calc(in_imp,feat_imp,lambda x,y:x*y)
    imp_cmb = imp_data2cmb(mul_imp)
    return imp_cmb[dict_ind]
#输入权重L1+输出权重L1
def eval_imp_l1norm2(model,binding_dicts,dict_ind=None,loader=None):
    out_weis = get_wights(binding_dicts,wei_from='out')
    in_weis=get_wights(binding_dicts,wei_from='in')
    out_imp = list_apply(out_weis, lambda x: np.average(np.abs(x), axis=(1, 2, 3)))
    in_imp = list_apply(in_weis, lambda x: np.average(np.abs(x), axis=(0, 2, 3)))
    add_imp = list_calc(in_imp, out_imp, lambda x, y: x + y)
    imp_cmb = imp_data2cmb(add_imp)
    return imp_cmb[dict_ind]
#特征图rank
def eval_imp_rank(model,binding_dicts,dict_ind=None,loader=None):
    batch_x, batch_y =next(iter(loader))
    in_maps = get_feat_maps(model, binding_dicts, batch_x, feat_from='out', rec='out')
    def rank_aver(data):
        b,c,_,_=data.shape
        rank_mat=np.zeros(shape=(b,c))
        for i in range(b):
            for j in range(c):
                s,v,dt=np.linalg.svd(data[i][j])
                rank=np.sum(v>=1e-5)
                rank_mat[i][j]=rank
        imp_rank=np.average(rank_mat,axis=0)
        return imp_rank
    imp_data=list_apply(in_maps[dict_ind:dict_ind+1],rank_aver)
    imp_cmb=imp_data2cmb(imp_data)
    return imp_cmb[0]

#特征图最大lamdba
def eval_imp_lambda(model,binding_dicts,dict_ind=None,loader=None):
    batch_x, batch_y = next(iter(loader))
    in_maps = get_feat_maps(model, binding_dicts, batch_x, feat_from='in', rec='in')
    def lambda_aver(data):
        b,c,_,_=data.shape
        rank_mat=np.zeros(shape=(b,c))
        for i in range(b):
            for j in range(c):
                s,v,dt=np.linalg.svd(data[i][j])
                rank_mat[i][j]=max(v)
        imp_rank=np.average(rank_mat,axis=0)
        return imp_rank
    imp_data=list_apply(in_maps[dict_ind:dict_ind+1],lambda_aver)
    imp_cmb=imp_data2cmb(imp_data)
    return imp_cmb[0]

#特征图中位数lamdba
def eval_imp_lambdaM(model,binding_dicts,dict_ind=None,loader=None):
    batch_x, batch_y = next(iter(loader))
    in_maps = get_feat_maps(model, binding_dicts, batch_x, feat_from='in', rec='in')
    def lambda_aver(data):
        b,c,_,_=data.shape
        rank_mat=np.zeros(shape=(b,c))
        for i in range(b):
            for j in range(c):
                s,v,dt=np.linalg.svd(data[i][j])
                rank_mat[i][j]=np.median(v)
        imp_rank=np.average(rank_mat,axis=0)
        return imp_rank
    imp_data=list_apply(in_maps[dict_ind:dict_ind+1],lambda_aver)
    imp_cmb=imp_data2cmb(imp_data)
    return imp_cmb[0]
#特征图std
def eval_imp_std(model,binding_dicts,dict_ind=None,loader=None):
    batch_x, batch_y =next(iter(loader))
    in_maps = get_feat_maps(model, binding_dicts, batch_x, feat_from='in', rec='in')
    def std_aver(data):
        b,c,_,_=data.shape
        std_mat=np.zeros(shape=(b,c))
        data=data.reshape(b,c,-1)
        data=np.std(data,axis=2)
        imp_std=np.average(data,axis=0)
        return imp_std
    imp_data=list_apply(in_maps[dict_ind:dict_ind+1],std_aver)
    imp_cmb=imp_data2cmb(imp_data)
    return imp_cmb[0]
#特征图APOZ
def eval_imp_apoz(model,binding_dicts,dict_ind=None,loader=None):
    batch_x, batch_y = next(iter(loader))
    in_maps = get_feat_maps(model, binding_dicts, batch_x, feat_from='in', rec='in')
    imp_data=list_apply(in_maps[dict_ind:dict_ind+1],lambda x:np.sum(x>=1e-5,axis=(0,2,3)))
    imp_cmb=imp_data2cmb(imp_data)
    return imp_cmb[0]
#####################################################



