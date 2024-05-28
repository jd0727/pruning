from tools import *
from models.praser import *
from models import *
from data import *
import copy
import pandas as pd
#####################
num_cls=10
train_loader = cifar3.get_train_loader(num_cls=num_cls, batch_size=128)
test_loader= cifar3.get_test_loader(num_cls=num_cls, batch_size=128)

prune_loader = cifar3.get_train_loader(num_cls=num_cls, batch_size=32)

file_name='..//chk//c10_vgg16'
data_name='..//chk//c10_vgg16_cert2//data.xlsx'
# model=resnetC(num_cls=num_cls,num_layer=20)
model=vggX(num_cls=10)
model.load_state_dict(torch.load(file_name))

#数据记录
data=pd.DataFrame(columns=['ep','flop','para','max acc','aver acc'])
flop_ori, param_ori=calc_flop_para(model,input_size=(32,32),ignore_zero=True)
acc = test_model(model, test_loader)
print('init acc ',acc)
data = data.append({
    'ep': 0, 'flop': flop_ori, 'para': param_ori,
    'max acc': acc, 'aver acc':acc
},ignore_index=True)
#开始剪枝
flop_rate=0.2
n=1
while True:
    print('ep ',n)
    # acc = test_model(model, test_loader)
    # print('before acc ',acc)
    #################
    # binding_dicts = ext_resnet(model, type='all2')
    binding_dicts = ext_vgg(model)
    # wei_out=get_wights(binding_dicts,wei_from='out')
    #通过特征图
    bx,_=next(iter(prune_loader))
    maps=get_feat_maps(model,binding_dicts,bx,feat_from='in',rec='in')
    maps = list_apply(maps, lambda x: x.reshape(x.shape[0], x.shape[1],-1))
    maps= list_apply(maps,lambda x:x.transpose(1,0,2))
    maps = list_apply(maps, lambda x: x.reshape(x.shape[0],  -1))
    ############
    # #L1剪枝
    # imp_cmb=eval_imp_l2norm(model,binding_dicts,dict_ind=None,loader=None)
    # #叠加lamp
    # imp_cmb=imp_cmb_lamp(imp_cmb)
    # imp_list=imp_cmb2list(imp_cmb)
    # #过滤掉imp=0，已经剪枝的
    # imp_list=[imp_msg for imp_msg in imp_list if imp_msg['imp']>1e-7]
    # #挑选10个
    # rmv_chan_list = rmv_list_by_order(imp_list, num=5)
    # rmv_chans(binding_dicts, rmv_chan_list,opr_type='frmv')
    ############
    #卷积核相似性剪枝
    sim_pairs = imp_by_sim(maps, thres=0, meth='pcos')
    sim_pairs = sim_pairs[:10]
    print(sim_pairs)
    cmb_chans(binding_dicts, sim_pairs, opr_type='fcmbp')
    #################
    # #综合评价
    # imp_cmb=eval_imp_rsg(model,binding_dicts,dict_ind=None,loader=prune_loader)
    # imp_list=imp_cmb2list(imp_cmb)
    # #过滤掉imp=0，已经剪枝的
    # imp_list=[imp_msg for imp_msg in imp_list if imp_msg['imp']>1e-7]
    # #挑选5个
    # rmv_chan_list = rmv_list_by_order(imp_list, num=5)
    # rmv_chans(binding_dicts, rmv_chan_list,opr_type='frmv')
    ##################################
    #格式调整
    # refmt_res(model)
    #################
    acc = test_model(model, test_loader)
    flop_cut, param_cut=calc_flop_para(model,input_size=(32,32),ignore_zero=False)
    #
    print('after acc ', acc)
    pre_rate=flop_cut/flop_ori
    print('prsev ', pre_rate)
    #
    model = copy.deepcopy(model)
    # full_name=os.path.join('../chk','tmp')
    # torch.save(model, full_name)
    # model = torch.load(full_name)
    #
    msg=train_SGD_StepLR(model, train_loader, test_loader, total_epoch=0, lr=0.01, momentum=0.9, weight_decay=5e-4,
                     milestones=[5], gamma=0.1, file_name='', file_dir='../', reg_loss=None, save_process=False)
    acc_seq=[m['acc'] for m in msg]
    if len(acc_seq)>0:
        data=data.append({
            'ep':n,'flop':flop_cut,'para':param_cut,
            'max acc':np.max(acc_seq),'aver acc':np.average(acc_seq[-5:])
        },ignore_index=True)
    #保存模型
    if abs(pre_rate*10-round(pre_rate*10))<0.1:
        torch.save(model.state_dict(), file_name+'_pre'+str(int(pre_rate*100)))
    #保存data记录
    if n%10==0:
        data.to_excel(data_name, index=False)

    if flop_rate>pre_rate:
        break
    else:
        n=n+1
    pass


# train_SGD_StepLR(model, train_loader, test_loader, total_epoch=40, lr=0.01, momentum=0.9, weight_decay=5e-4,
#                  milestones=[10], gamma=0.1, file_name='', file_dir='../', reg_loss=None, save_process=False)
# t_save(models,0,0,0,file_name=file_name+'_half',file_dir='../chk')


