from tools import *
from models.praser import *
from models import *
from data import *
import torch
import numpy as np

# wei=torch.as_tensor(weis[dict_ind][0])
# feat=torch.as_tensor(feats[dict_ind][0])
# scores=np.zeros(shape=wei.shape[0])
# for i in range(wei.shape[0]):
#     fake_fliter=torch.zeros(size=(wei.shape[1],wei.shape[1],3,3))
#     for j in range(wei.shape[1]):
#         fake_fliter[j,j]=wei[i,j]
#     piece=F.conv2d(feat,fake_fliter,stride=1,padding=0)
#     piece=piece.detach().cpu().numpy()
#     piece=np.transpose(piece,(1,0,2,3))
#     piece=piece.reshape(wei.shape[1],-1)
#     l2=np.sum(piece*piece,axis=1)
#     #排序
#     order=np.argsort(l2)
#     sum_vec=np.sum(piece,axis=0)
#     sum_norm=np.linalg.norm(sum_vec)
#     acc_vec=0
#     for k in range(wei.shape[1]):
#         acc_vec=acc_vec+piece[order[k]]
#         acc_norm = np.linalg.norm(acc_vec)
#         if acc_norm>sum_norm*0.2:
#             break
#     presv_inds=order[(k-1):]
#     scores[presv_inds]+=1


def eval_imp_rs(model,binding_dicts,dict_ind=0,loader=None,rate=10):
    grads=[]
    weis=get_wights(binding_dicts,wei_from='out')

    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()

    optimizer=torch.optim.SGD(model.parameters(),lr=0.1, momentum=0.9, weight_decay=5e-4)
    for i,(batch_x,batch_y) in enumerate(loader):
        if torch.cuda.is_available():
            batch_x = batch_x.cuda()
            batch_y=batch_y.cuda()

        batch_yp = model(batch_x)
        criterion = nn.CrossEntropyLoss().cuda()

        loss=criterion(batch_yp,batch_y)
        loss.backward()
        #记录权重
        w_i = weis[dict_ind][0]
        grad = w_i.grad.detach().cpu().numpy()
        grads.append(grad)
        #
        optimizer.zero_grad()
        if i==127:
            break

    grads=np.stack(grads,axis=0)
    grads_std=np.std(grads,axis=0)

    vals=weis[dict_ind][0].data.detach().cpu().numpy()
    out_channs=vals.shape[0]

    vals=vals.reshape(out_channs,-1)
    grads_std=grads_std.reshape(out_channs,-1)


    rs=np.linalg.norm(grads_std,axis=1)
    rvs=np.linalg.norm(vals, axis=1)

    order=np.argsort(rvs)
    acc_rs=np.sum(rs)
    acc_rps=0

    rmv_inds=None
    for i in range(out_channs):
        ind=order[i]
        acc_rs-=rs[ind]
        acc_rps+=rvs[ind]
        if acc_rs*rate<acc_rps:
            rmv_inds=order[:i]
            break

    cmb_inds=[]
    cover_mat=np.zeros(shape=(out_channs,out_channs))
    for i in range(out_channs):
        if i in rmv_inds:
            continue
        for j in range(i):
            if j in rmv_inds:
                continue
            dist=np.linalg.norm(vals[i]-vals[j])
            cover=rs[i]+rs[j]
            cover_mat[i,j]=cover*rate/dist

    while True:
        i, j = np.where(cover_mat == np.max(cover_mat))
        i=i[0]
        j=j[0]
        if cover_mat[i,j]<1:
            break
        else:
            cmb_inds.append([i,j])
            cover_mat[i, :] = 0
            cover_mat[j, :] = 0
            cover_mat[:, i] = 0
            cover_mat[:, j] = 0
    return rmv_inds,cmb_inds

#适用于全局剪枝的评价方法
def eval_imp_rsg(model,binding_dicts,dict_ind=0,loader=None):
    weis = get_wights(binding_dicts,wei_from='out')
    grads = [[] for _ in range(len(weis))]

    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()

    optimizer=torch.optim.SGD(model.parameters(),lr=0.1, momentum=0.9, weight_decay=5e-4)
    for i,(batch_x,batch_y) in enumerate(loader):
        if torch.cuda.is_available():
            batch_x = batch_x.cuda()
            batch_y=batch_y.cuda()

        batch_yp = model(batch_x)
        criterion = nn.CrossEntropyLoss().cuda()

        loss=criterion(batch_yp,batch_y)
        loss.backward()
        #记录梯度
        for i in range(len(grads)):
            grads[i].append(weis[i][0].grad.detach().cpu().numpy())
        #
        optimizer.zero_grad()
        if i==31:
            break
    imp_cmbs=[]
    for i in range(len(grads)):
        grads_i=np.array(grads[i])
        grads_i= np.stack(grads_i,axis=0)
        grads_i = np.std(grads_i, axis=0)
        out_channs = grads_i.shape[0]
        grads_i = grads_i.reshape(out_channs, -1)
        grads_i = np.average(grads_i, axis=1)
        #
        wei_i = weis[i][0].data.detach().cpu().numpy()
        wei_i = wei_i.reshape(out_channs, -1)
        wei_i = np.sqrt(np.average(wei_i ** 2, axis=1))
        # 归一化
        # wei_i = wei_i / np.sum(wei_i)
        # grads_i = grads_i / np.sum(grads_i)
        #
        grads_i[grads_i==0]=1
        imp_cmbs.append(wei_i/grads_i)

    return imp_cmbs

if __name__ == '__main__':
    prune_loader = get_train_loader(num_cls=10, batch_size=32)
    #
    model=resnetC(num_cls=10,num_layer=20,act='relu')
    model.load_state_dict(torch.load('../chk/c10_res20'))
    binding_dicts = ext_resnet(model, meth='all2', input_size=(32, 32))
    # imp_cmbs=eval_imp_rsg(model, binding_dicts, None ,prune_loader)
    weis = get_wights(binding_dicts, wei_from='out')
    # imps=eval_imp_rs(model,binding_dicts,0,prune_loader)
    # weis = get_wights(binding_dicts, wei_from='out')
    pairs=imp_by_sim(weis,meth='pcos',thres=0)
    pairs=pairs[:5]
    # cmb_chans(binding_dicts, pairs, opr_type='fcmbp')
    # #
    # flop_cut, param_cut = calc_flop_para(model, input_size=(32, 32), ignore_zero=False)

    # imps_cmbs2=eval_imp_l1norm(model, binding_dicts, None)
    # for p in pairs:
    #     dict_ind=p['dict_ind']
    #     ind1,ind2=p['chan_inds']
    #     print(imps_cmbs2[dict_ind][ind1],imps_cmbs2[dict_ind][ind2],'sim',p['sim'],'aver',np.average(imps_cmbs2[dict_ind]))

    # for b in binding_dicts:
    #     print(len(b['in']),len(b['pth']),len(b['out']))


#
#
# # flop_ori, param_ori=calc_flop_para(model,input_size=(32,32),ignore_zero=False)
#
#






