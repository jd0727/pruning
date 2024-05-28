from data import *
from tools import *
from models import *
from models.praser import *
from sklearn.cluster import KMeans,AffinityPropagation


##########################################################################kmeans聚类判别
def cluster_by_kmeans(data, n_clusters=3):
    estimator = KMeans(n_clusters=n_clusters)  # 构造聚类器
    estimator.fit(data)  # 聚类
    label = estimator.labels_
    label = np.array(label)
    clusters = []
    for i in range(n_clusters):
        cls_inds = np.nonzero(label == i)[0]
        clusters.append(cls_inds)
    return clusters
# clust
def eval_imp_cluster(model, binding_dicts, dict_ind=None, loader=None):
    out_weis = get_wights(binding_dicts, wei_from='out')
    w = out_weis[dict_ind][0]
    w = w.reshape(w.shape[0], -1)
    l2norm = np.sum(w * w, axis=1)
    n_clusters = 3
    clusters = cluster_by_kmeans(w, n_clusters=n_clusters)
    imps = np.zeros(shape=w.shape[0])
    for i in range(n_clusters):
        imps_i = l2norm[clusters[i]]
        imps_i = imps_i / np.sum(imps_i)
        imps[clusters[i]] = imps_i

    return imps
##########################################################################NMS
def nms(w,thres=0.5):
    w = w.reshape(w.shape[0], -1)
    sim_mat = np.abs(cos_sim_mat(w))
    l2norm=np.sum(w*w,axis=1)
    #
    order=np.argsort(l2norm)
    order=order[::-1]#降序
    sim_mat=sim_mat[order]
    #
    presv=[]
    label=np.ones(shape=w.shape[0])
    for i in range(w.shape[0]):
        if label[i]==0:
            continue
        over_inds=sim_mat[i]>thres
        label[over_inds]=0
        presv.append(order[i])
    #
    return presv
#NMS
def eval_imp_nms(model,binding_dicts,dict_ind=None,loader=None):
    out_weis = get_wights(binding_dicts,wei_from='out')
    w = out_weis[dict_ind][0]
    w=w.reshape(w.shape[0], -1)
    presv=nms(w,thres=0.4)
    return presv
##########################################################################AP聚类判别
def eval_imp_ap(model,binding_dicts,dict_ind=None,loader=None):
    #特定层
    binding_dicts=binding_dicts[dict_ind:dict_ind+1]
    # 获取特征图
    bx,_=next(iter(loader))
    maps= get_feat_maps(model,binding_dicts,bx,feat_from='in',rec='in')
    maps = list_apply(maps, lambda x: x.reshape(x.shape[0], x.shape[1],-1))
    maps= list_apply(maps,lambda x:x.transpose(1,0,2))
    maps = list_apply(maps, lambda x: x.reshape(x.shape[0],  -1))
    #提取相似度矩阵
    sim_mat=get_sim_mat(maps,meth='pcos')
    sim_mat=sim_mat[0]
    #聚类
    af = AffinityPropagation(random_state=0, affinity='precomputed')
    af.fit_predict(sim_mat)
    cluster_centers_indices = af.cluster_centers_indices_
    labels = af.labels_
    return cluster_centers_indices
##########################################################################连接权重判别
def eval_imp_pcos(model,binding_dicts,dict_ind=None,loader=None,thres=0.1):
    #特定层
    binding_dicts=binding_dicts[dict_ind:dict_ind+1]
    # 获取特征图
    bx,_=next(iter(loader))
    maps= get_feat_maps(model,binding_dicts,bx,feat_from='in',rec='in')
    maps = list_apply(maps, lambda x: x.reshape(x.shape[0], x.shape[1],-1))
    maps= list_apply(maps,lambda x:x.transpose(1,0,2))
    maps = list_apply(maps, lambda x: x.reshape(x.shape[0],  -1))
    #提取相似度矩阵
    sim_pairs = imp_by_sim(maps, thres=thres, meth='pcos')
    for i in range(len(sim_pairs)):#还原为整体序号
        sim_pairs[i]['dict_ind']=dict_ind
    return sim_pairs

##########################################################################实验
if __name__ == '__main__':
    num_cls = 10
    train_loader = cifar.get_train_loader(num_cls=num_cls, batch_size=128)
    test_loader = cifar.get_test_loader(num_cls=num_cls, batch_size=128)
    prune_loader = cifar.get_train_loader(num_cls=num_cls, batch_size=32)

    model = vggX(num_cls=num_cls)
    # model.load_state_dict(torch.load('../chk/c10_vgg16'))
    binding_dicts = ext_vgg(model)

    # model=resnetC(num_cls=10,num_layer=20)
    # model.load_state_dict(torch.load('../chk/c10_res20'))

    prs=eval_imp_pcos(model,binding_dicts,dict_ind=1,loader=prune_loader,thres=0.1)


