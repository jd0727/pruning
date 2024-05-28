import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import copy
#######################################################得到feature_map
#按类别平均
def ave_by_cls(feat_maps,batch_y,num_cls=10):
    if isinstance(batch_y,torch.Tensor):
        batch_y=batch_y.detach().cpu().numpy()
    #按lb分类
    cls_distr = np.zeros(shape=num_cls)
    for i, lb in enumerate(batch_y):
        cls_distr[lb] += 1

    ave_feat_maps=[]
    for i,maps in enumerate(feat_maps):
        ave_feat_maps.append([])
        for j,map in enumerate(maps):
            shape = list(map.shape)
            shape[0]=num_cls
            ave_feat_map=np.zeros(shape)
            #求和
            for k,lb in enumerate(batch_y):
                ave_feat_map[lb]+=map[k]
            #均值
            for r in range(num_cls):
                if cls_distr[r]==0:
                    print('err no cls')
                    continue
                ave_feat_map[r]=ave_feat_map[r]/cls_distr[r]
            ave_feat_maps[i].append(ave_feat_map)
    return ave_feat_maps

#按batch平均
def ave_by_batch(feat_maps):
    ave_feat_maps = []
    for i,maps in enumerate(feat_maps):
        ave_feat_maps.append([])
        for j,map in enumerate(maps):
            ave_feat_maps[i].append(np.average(map,axis=0))
    return ave_feat_maps

#得到batch的featmap
def get_feat_maps(model, binding_dicts, batch_x,feat_from='in',rec='in'):
    #init buffer
    handles = []
    feat_maps = []
    for i, binding_dict in enumerate(binding_dicts):
        feat_maps.append([None]*len(binding_dict[feat_from]))
    #hook def
    def get_recorder(ind_dict,ind_mod):
        if rec=='in':
            def recorder(model, input, output):
                input = input[0]
                feat_maps[ind_dict][ind_mod] = input
                return
            return recorder
        elif rec=='out':
            def recorder(model, input, output):
                feat_maps[ind_dict][ind_mod] = output
                return
            return recorder
        else:
            print('err type',rec)
            return None
    #添加hook
    for i, binding_dict in enumerate(binding_dicts):
        for j,sub_model in enumerate(binding_dict[feat_from]):
            handle = sub_model.register_forward_hook(get_recorder(i,j))
            handles.append(handle)
    # 传播1个batch样本
    device=next(iter(model.parameters())).device

    # device=model.device
    batch_x=batch_x.to(device)
    output = model(batch_x)
    #清除hook
    for hook in handles:
        hook.remove()
    return feat_maps

#得到loader所有的feat叠加
def get_ave_maps_all(model, binding_dicts, loader, num_cls=10):
    lb=None
    handles = []
    cls_num_sum=np.zeros(shape=num_cls)
    ave_feat_maps = []
    for i, binding_dict in enumerate(binding_dicts):
        ave_feat_maps.append([None]*len(binding_dict['in']))

    #hook def
    def get_hook_input(ind_dict,ind_mod):
        def recorder(model, input, output):
            if isinstance(input,tuple):
                input = input[0]

            input = input.cpu().detach().numpy()
            shape = list(input.shape)
            shape[0] = num_cls
            tmp_sum = np.zeros(shape=shape)
            #叠加
            for i,lb in enumerate(lbs):
                tmp_sum[lb]+=input[i]
            #记录
            if ave_feat_maps[ind_dict][ind_mod] is None:
                ave_feat_maps[ind_dict][ind_mod]=tmp_sum
            else:
                ave_feat_maps[ind_dict][ind_mod] += tmp_sum
        return recorder

    #添加hook
    for i, binding_dict in enumerate(binding_dicts):
        for j,sub_model in enumerate(binding_dict['in']):
            handle = sub_model.register_forward_hook(get_hook_input(i,j))
            handles.append(handle)

    num_iter = 0
    if torch.cuda.is_available():
        model = model.cuda()
    for batch_x,batch_y in loader:
        lbs=batch_y.detach().cpu().numpy()
        for i, lb in enumerate(lbs):
            cls_num_sum[lb] += 1
        # 传播1个batch样本
        if torch.cuda.is_available():
            batch_x = batch_x.cuda()
        output = model(batch_x)
        #统计
        num_iter+=1
        if num_iter%100==0:
            print('iter ',num_iter)

    #均值处理
    for i in range(len(ave_feat_maps)):
        for j in range(len(ave_feat_maps[i])):
            feat_sum =ave_feat_maps[i][j]
            for k in range(num_cls):
                feat_sum[k] /= cls_num_sum[k]
            ave_feat_maps[i][j]=feat_sum

    #清除hook
    for hook in handles:
        hook.remove()

    return ave_feat_maps


#######################################################得到模型权重
#得到权重信息
def get_wights(binding_dicts,wei_from='out'):
    wei_paras = []
    for i, binding_dict in enumerate(binding_dicts):
        wei_paras.append([None]*len(binding_dict[wei_from]))
    #创建值
    for i, binding_dict in enumerate(binding_dicts):
        for j,sub_model in enumerate(binding_dict[wei_from]):
            # wei_paras[i][j]=sub_model.weight.data.detach().cpu().numpy()
            wei_paras[i][j] = sub_model.weight
    return wei_paras
#得到偏移量信息
def get_bias(binding_dicts,wei_from='out'):
    bias_paras = []
    for i, binding_dict in enumerate(binding_dicts):
        bias_paras.append([None]*len(binding_dict[wei_from]))
    #创建值
    for i, binding_dict in enumerate(binding_dicts):
        for j,sub_model in enumerate(binding_dict[wei_from]):
            if not sub_model.bias is None:
                bias_paras[i][j]=sub_model.bias.data.detach().cpu().numpy()
    return bias_paras
#######################################################计算cos相似度

#计算cos相似矩阵
def cos_sim_mat(fliters):
    fliters = fliters.reshape(fliters.shape[0], -1)
    sim_mat = cosine_similarity(fliters)
    for i in range(fliters.shape[0]):
        sim_mat[i, i] = 0
    return sim_mat

def corr_sim_mat(fliters):
    fliters = fliters.reshape(fliters.shape[0], -1)
    sim_mat=np.corrcoef(fliters)
    for i in range(fliters.shape[0]):
        sim_mat[i, i] = 0
    return sim_mat
#得到任意度量相似矩阵
def get_sim_mat(paras,meth='cos'):
    sim_mats=[]
    for i,para in enumerate(paras):
        #规范输入
        para=para[0]
        if isinstance(para,torch.Tensor) or isinstance(para,torch.nn.Parameter):
            para=para.data.detach().cpu().numpy()
        #整形
        if isinstance(para,np.ndarray):
            para=np.reshape(para,newshape=(para.shape[0],-1))
        else:
            raise Exception('type err')
        #度量
        if meth=='cos':
            sim_mat = cosine_similarity(para)
            sim_mat = np.abs(sim_mat)
            for i in range(sim_mat.shape[0]):
                sim_mat[i, i] = 0
        elif meth == 'negl2':
            m, n = para.shape
            G = np.dot(para.T, para)
            H = np.tile(np.diag(G), (n, 1))
            sim_mat=-(H + H.T - 2*G)
        elif  meth=='corr':
            sim_mat = np.corrcoef(para)
            for i in range(sim_mat.shape[0]):
                sim_mat[i, i] = 0
        elif  meth=='p':
            pows = np.sqrt(np.average(para ** 2, axis=1))
            pows = pows / np.average(pows)#归一化
            p1,p2 = np.meshgrid(pows,pows)
            fac=np.power(1/p1/p2,1)
            for i in range(fac.shape[0]):
                fac[i, i] = 0
            sim_mat=fac
        elif  meth=='pcos':
            #排除已剪枝通道的影响
            pows = np.sqrt(np.average(para ** 2, axis=1))
            num_chans=pows.shape[0]
            val_inds=(pows>0.001)
            val_inds = np.nonzero(val_inds)[0]
            pows=pows[val_inds]
            para=para[val_inds]
            #计算权重
            sim_mat_p = cosine_similarity(para)
            sim_mat_p = np.abs(sim_mat_p)
            #实验1
            pows=pows/np.average(pows)#归一化
            p1,p2 = np.meshgrid(pows,pows)
            fac=np.power(1/(p1*p2+0.1),1)
            sim_mat_p= sim_mat_p*fac
            for k in range(len(pows)):
                sim_mat_p[k,k]=0
            #还原成原矩阵
            sim_mat=np.zeros(shape=(num_chans,num_chans))
            x, y = np.meshgrid(val_inds, val_inds)
            sim_mat[x,y]=sim_mat_p
        else:
            raise Exception('meth err')
        sim_mats.append(sim_mat)
    return sim_mats

#从sim_mat得到pairs
def get_pairs(sim_mats,thres=0.001):
    #Pairs
    sim_pairs = []
    for i, sim_mat in enumerate(sim_mats):
        for ind1 in range(sim_mat.shape[0]):
            for ind2 in range(ind1):
                if sim_mat[ind1,ind2]<thres:
                    continue
                sim_pairs.append({
                    'dict_ind': i,
                    'chan_inds': [ind1, ind2],
                    'sim': sim_mat[ind1, ind2]
                })
    #NMS
    sim_pairs = sorted(sim_pairs, key=lambda dic: dic['sim'])
    sim_pairs.reverse()
    ind_set=[]
    sim_pairs_nms=[]
    for i in range(len(sim_pairs)):
        dict_ind = sim_pairs[i]['dict_ind']
        ind1,ind2 = sim_pairs[i]['chan_inds']
        if (dict_ind,ind1) in ind_set or (dict_ind,ind2) in ind_set:
            continue
        else:
            ind_set.append((dict_ind,ind1))
            ind_set.append((dict_ind, ind2))
            sim_pairs_nms.append(sim_pairs[i])
    return sim_pairs_nms

#通过权重相似度判别
def imp_by_sim(paras, thres=0.001,meth='cos'):
    sim_mats = get_sim_mat(paras,meth)
    sim_pairs = get_pairs(sim_mats,thres)
    return sim_pairs
#######################################################对列表结构施加变换
#对列表嵌套结构进行变换
def list_apply(datas, func=None):
    if isinstance(datas, list):
        datas_r=[]
        for d in datas:
            datas_r.append(list_apply(d, func))
        return datas_r
    else:
        return func(datas)
#两个list结构计算
def list_calc(datas1,datas2, func=None):
    if isinstance(datas1, list):
        assert isinstance(datas2, list),'should be list'
        assert len(datas1)==len(datas2),'should same len'
        datas_r=[]
        for i in range(len(datas1)):
            datas_r.append(list_calc(datas1[i], datas2[i],func))
        return datas_r
    else:
        return func(datas1,datas2)

#生成具有同样嵌套结构的空列表
def list_like(datas,val=None):
    if isinstance(datas, list):
        data_r=[]
        for d in datas:
            data_r.append(list_like(d,val=val))
        return data_r
    else:
        return val
#######################################################
#计算序列余弦相似
def cos_sim(arr1,arr2):
    norm1=np.linalg.norm(arr1, ord=2)
    norm2 = np.linalg.norm(arr2, ord=2)
    cos=np.dot(arr1, arr2)/norm1/norm2
    return cos

#归一化
def imp_cmb_unit(imp_cmb):
    for i in range(len(imp_cmb)):
        max_val=np.max(imp_cmb[i])
        min_val = np.min(imp_cmb[i])
        imp_cmb[i]=(imp_cmb[i]-min_val)/(max_val-min_val)
    return imp_cmb

#计算LAMP
def imp_cmb_lamp(imp_cmb):
    for i in range(len(imp_cmb)):
        order=np.argsort(imp_cmb[i])
        order=order[::-1]
        size=len(imp_cmb[i])
        imp_cmb_new=np.zeros(size)
        sum_val=0
        for j in range(size):
            sum_val+=imp_cmb[i][order[j]]
            imp_cmb_new[order[j]]=imp_cmb[i][order[j]]/sum_val
        imp_cmb[i]=imp_cmb_new
    return imp_cmb



#######################################################格式转换
#imp_data=[[conv1][conv1,conv2]...]
#imp_cmb=[[chan_imp0][chan_imp1]...]
#imp_list=[{bind_ind,chan_ind,imp}{}...]

#imp_data转cmb
def imp_data2cmb(imp_data):
    imp_cmb=[]
    for i,bind_data in enumerate(imp_data):
        aver_imp=0
        for j, chan_data in enumerate(bind_data):
            aver_imp+=chan_data
        aver_imp/=len(bind_data)
        imp_cmb.append(aver_imp)
    return imp_cmb

#imp_list转cmb
def imp_list2cmb(imp_list):
    max_dict=0
    for imp in imp_list:
        max_dict=max(max_dict,imp['dict_ind'])
    max_chans=np.zeros(shape=(max_dict+1))
    for imp in imp_list:
        max_chans[imp['dict_ind']]=max(max_dict,imp['chan_ind'])
    imp_cmb= [None]*(max_dict+1)
    for imp in imp_list:
        dict_ind=imp['dict_ind']
        chan_ind=imp['chan_ind']
        if dict_ind not in imp_cmb.keys():
            imp_cmb[dict_ind]=np.zeros(shape=int(max_chans[dict_ind]+1))
        imp_cmb[dict_ind][chan_ind]=imp['imp']
    return imp_cmb

#cmb转list
def imp_cmb2list(imp_cmb):
    imp_list=[]
    for i, bind_data in enumerate(imp_cmb):
        for j,chan_imp in enumerate(bind_data):
            imp_list.append({
                'dict_ind':i,
                'chan_ind':j,
                'imp':chan_imp
            })
    return imp_list
#######################################################
