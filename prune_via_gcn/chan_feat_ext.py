from tools import *
from data import *
from models import *



#从多个特征图中提取分布，并组合
def ext_maps_distri(maps,max_val=3,div_num = 12):
    map_distris = []
    for map in maps:
        if isinstance(map,torch.Tensor):
            map=map.detach().cpu().numpy()
        map = map.reshape(map.shape[0], map.shape[1], -1)  # (N,C,F)
        map = map.transpose(1, 0, 2)  # (C,N,F)
        map = map / np.std(map)  # 归一化
        map_mean = np.average(map, axis=2)  # 特征图均值池化 # (C,N)
        step = max_val / div_num
        # 量化
        map_mean = np.floor(map_mean / step)
        # 计数
        map_distri = [np.sum(map_mean == k, axis=1) for k in range(div_num)]
        map_distri = np.stack(map_distri, axis=1)
        map_distri = map_distri / map_mean.shape[1]
        map_distris.append(map_distri)
    map_distris = np.concatenate(map_distris, axis=1)
    map_distris=map_distris/np.sum(map_distris,axis=1,keepdims=True)
    return map_distris

#从BN中提取特征并组合
def ext_bn(weis, baiss):
    assert len(weis)==len(baiss),'len err'
    bn_fmts=[]
    for i in range(len(weis)):
        wei=weis[i]
        bais=baiss[i]
        if isinstance(wei, torch.Tensor):
            wei = wei.detach().cpu().numpy()
        wei = wei.reshape(wei.shape[0], -1)
        # 偏置处理
        if isinstance(bais, torch.Tensor):
            bais = bais.detach().cpu().numpy()
        bais = bais.reshape(bais.shape[0], -1)
        bn_fmt=np.concatenate([wei,bais], axis=1)
        bn_fmts.append(bn_fmt)
    bn_fmts=np.concatenate(bn_fmts, axis=1)
    return bn_fmts

#带BN的
def ext_feats_bn(model, binding_dicts, bx, by):
    mapss = get_feat_maps(model, binding_dicts, bx, feat_from='in', rec='in')
    weiss = get_wights(binding_dicts, wei_from='pth')
    baiss = get_bias(binding_dicts, wei_from='pth')
    feats = []
    for i in range(len(binding_dicts)):
        map_distris=ext_maps_distri(mapss[i],max_val=3,div_num = 12)
        bn_fmts=ext_bn(weiss[i],baiss[i])
        feat=np.concatenate([map_distris,bn_fmts], axis=1)
        feats.append(feat)
    return feats

#不带bn的
def ext_feats_map(model, binding_dicts, bx, by,max_val=3,div_num = 12):
    mapss = get_feat_maps(model, binding_dicts, bx, feat_from='in', rec='in')
    feats = []
    for i in range(len(binding_dicts)):
        map_distris=ext_maps_distri(mapss[i],max_val=max_val,div_num = div_num)
        feats.append(map_distris)
    return feats


#################################################################################计算边
def get_dist_mat(feats,meth='l2'):
    assert len(feats.shape)==2,'shape err'#(N,F)
    m, n = feats.shape
    if meth == 'l2':
        G = np.dot(feats, feats.T)
        H = np.tile(np.diag(G), (m, 1))
        dist_mat = np.sqrt(H + H.T - 2 * G)
    elif meth=='kl':
        P = np.expand_dims(feats, 1).repeat(m, axis=1)
        Q = np.expand_dims(feats, 0).repeat(m, axis=0)
        E=P*(np.log(P+1e-5)-np.log(Q+1e-5))
        E=np.sum(E,axis=2)
        dist_mat=E
    else:
        print('Err meth',meth)
        return None
    return dist_mat

# 通过相似度矩阵得到边
def get_edges(dist_mat, k=5):
    num = dist_mat.shape[0]
    es = []
    for i in range(num):
        order = np.argsort(dist_mat[i, :])
        order = order[:k]
        for n in range(k):
            if not (order[n], i) in es:
                es.append((i, order[n]))
    es=np.array(es)
    return es

def get_edges_rate(dist_mat,rate=0.2):
    num = dist_mat.shape[0]
    thres=np.sqrt(np.average(dist_mat**2))*rate
    for i in range(num):
        dist_mat[i,i]=dist_mat[i,i]+thres*2#防止自环
    es=np.nonzero(dist_mat<thres)
    es=np.stack(es,axis=1)
    return es

def get_edges_thres(dist_mat,thres=0.2):
    num = dist_mat.shape[0]
    for i in range(num):
        dist_mat[i,i]=dist_mat[i,i]+thres*2#防止自环
    es=np.nonzero(dist_mat<thres)
    es=np.stack(es,axis=1)
    return es

if __name__ == '__main__':
    num_cls=100
    test_loader = cifar_loader(set='test', num_cls=num_cls, batch_size=256)
    train_loader = cifar_loader(set='train', num_cls=num_cls, batch_size=256)
    #
    # # 模型
    model = CLSframe(backbone='vgg', num_cls=num_cls)
    model.load_wei('../chk/c100_vgg16')

    # model = CLSframe(backbone='resC', num_cls=num_cls,num_layer=56)
    # model.load_wei('../chk/c10_res56')

    binding_dicts = model.get_dicts(meth='all')
    # binding_dicts = binding_dicts[11:12]
    # #统计
    bx, by = next(iter(train_loader))
    feats=ext_feats_map(model, binding_dicts, bx, by)

    for i in range(15):
        dist_mat=get_dist_mat(feats[i],meth='kl')
        val=np.mean(dist_mat)
        print('V',val)
        es=get_edges_thres(dist_mat,thres=0.05)
        print(len(es)/len(dist_mat)**2)
        # porder_arr(datas=feats[i][:36],shower=show_curve)

    # es=edges_thres(dist_mat,rate=0.4)
