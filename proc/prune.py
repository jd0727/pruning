from tools import *
from models.praser import *
# from models import *
# from data import *
import torch


#模型剪枝
def prune_model(model, train_loader, test_loader,input_size=(32, 32), ext_type='all',
                file_name='prune', file_dir='../',imp_type='func',
                milestones_proc=None, gamma_proc=0.1, epoch_proc=10,lr_proc=0.1,
                milestones_rec=None,gamma_rec=0.1, epoch_rec=20,lr_rec=0.1,**kwargs):
    #规范输入
    if milestones_proc is None:
        milestones_proc = [3, 5]
    if milestones_rec is None:
        milestones_rec = [5, 10]
    #开始剪枝
    flop_ori, param_ori = calc_flop_param(model, input_size=input_size)
    binding_dicts = ext_resnet(model, meth=ext_type)
    num_dict = len(binding_dicts)
    acc_ori = test_model(model, test_loader)
    prsv_indss = {}
    for dict_ind in range(num_dict):
        print('pruning dict ', dict_ind)
        acc = test_model(model, test_loader)
        print('before acc ', acc)
        #################通过自定义函数评价
        if imp_type=='func':
            eval_imp=kwargs['eval_imp']
            ratio=kwargs['ratio']
            binding_dicts = ext_resnet(model, meth=ext_type)
            imps = eval_imp(model,binding_dicts,dict_ind=dict_ind,loader=train_loader)
            size = binding_dicts[dict_ind]['size']
            assert size == len(imps), 'need to be same'
            order = np.argsort(imps)
            prsv_inds = order[(size - int(size * ratio)):]
            prsv_indss[dict_ind] = prsv_inds
            chan_opr(binding_dicts[dict_ind], inds=prsv_inds, opr_type='ext')
        #################通过给定的索引移除
        elif imp_type=='rmv':
            binding_dicts = ext_resnet(model, meth=ext_type)
            rmv_indss = kwargs['rmv_indss']
            chan_opr(binding_dicts[dict_ind], inds=rmv_indss[dict_ind], opr_type='rmv')
        #################
        else:
            print('err type',imp_type)
        # 格式调整
        refmt_res(model)
        #################
        acc = test_model(model, test_loader)
        flop_cut, param_cut = calc_flop_param(model, input_size=(32, 32))
        print('after acc ', acc)
        print('prsev ', flop_cut / flop_ori)
        #
        full_name=os.path.join('../chk',file_name)
        torch.save(model,full_name)
        model=torch.load(full_name)
        #
        train_SGD_StepLR(model, train_loader, test_loader, total_epoch=epoch_proc, lr=lr_proc, momentum=0.9, weight_decay=5e-4,
                         milestones=milestones_proc, gamma=gamma_proc, file_name=file_name, file_dir=file_dir,
                         reg_loss=None, save_process=False)
        print('recv end')

    # 恢复训练
    msg=train_SGD_StepLR(model, train_loader, test_loader, total_epoch=epoch_rec, lr=lr_rec, momentum=0.9, weight_decay=5e-4,
                     milestones=milestones_rec, gamma=gamma_rec, file_name=file_name, file_dir=file_dir,
                     reg_loss=None, save_process=False)
    acc_cut = test_model(model, test_loader)
    flop_cut, param_cut = calc_flop_param(model, input_size=(32, 32))
    #
    msg_dict={'flop_cut':flop_cut,
              'param_cut':param_cut,
              'flop_ori':flop_ori,
              'param_ori':param_ori,
              'acc_final':acc_cut,
              'acc_ori':acc_ori,
              'prsv_indss':prsv_indss,
              'file_name':file_name,
              'msg_rec':msg}

    return model,msg_dict

if __name__ == '__main__':
    train_loader = cifar.get_train_loader(num_cls=10, batch_size=128)
    test_loader = cifar.get_test_loader()

    # models=resnet_cifar.resnetC(num_layer=20,num_cls=10)
    file_name = 'c10_res56'
    # file_name='c10_res20'
    model, num_epoch, num_iter, acc = t_load(file_name, file_dir='../chk')

    # binding_dicts=ext_all_chan(models,type='inner')
    # dict_ind=0
    # imp_f=eval_imp_byfeat(models,binding_dicts,dict_ind)
    # imp_ff=eval_imp_byff(models,binding_dicts,dict_ind)
    # imp_l=eval_imp_l1norm(models,binding_dicts,dict_ind)
    # imp_l2=eval_imp_l1norm2(models,binding_dicts,dict_ind)
    #
    # tx=list_calc(imp_l,imp_l2,cos_sim)

    # models,msg=prune_model(models,train_loader,test_loader,eval_imp=eval_imp_byfeat,
    #             input_size=(32, 32),ratio=0.8,ext_type='inner',file_name='c10_res20_in80_feat')
    # models,msg=prune_model(models,train_loader,test_loader,eval_imp=eval_imp_byff,
    #             input_size=(32, 32),ratio=0.8,ext_type='inner',file_name='c10_res20_in80_ff')
    # models,msg=prune_modelL(models, train_loader, test_loader, eval_imp=eval_imp_l1norm,
    #                        input_size=(32, 32), ratio=0.8, ext_type='inner', file_name='c10_res20_lk_in80_l1_30')
    # models,msg=prune_model(models,train_loader,test_loader,eval_imp=eval_imp_rank,
    #             input_size=(32, 32),ratio=0.8,ext_type='inner',file_name='c10_res20_rank')
    # models,msg=prune_model(models,train_loader,test_loader,eval_imp=eval_imp_lambda,
    #             input_size=(32, 32),ratio=0.8,ext_type='inner',file_name='c10_res20_lambda')
    # models,msg=prune_model(models,train_loader,test_loader,eval_imp=eval_imp_std,
    #             input_size=(32, 32),ratio=0.8,ext_type='inner',file_name='c10_res20_std')
    # models,msg=prune_model(models,train_loader,test_loader,eval_imp=eval_imp_lambdaM,
    #             input_size=(32, 32),ratio=0.8,ext_type='inner',file_name='c10_res20_lambdaM')
    # models,msg=prune_model(models,train_loader,test_loader,eval_imp=eval_imp_apoz,
    #             input_size=(32, 32),ratio=0.8,ext_type='inner',file_name='c10_res20_apoz')

    # models,msg=prune_model(models,train_loader,test_loader,eval_imp=eval_imp_lambda,
    #             input_size=(32, 32),ratio=0.8,ext_type='inner',file_name='c10_res56_lambda',
    #                       milestones_proc=[10],epoch_proc=30,lr_proc=0.01,
    #                       epoch_rec=0)
    # models,msg=prune_model(models,train_loader,test_loader,eval_imp=eval_imp_lambdaM,
    #             input_size=(32, 32),ratio=0.8,ext_type='inner',file_name='c10_res56_lambdaM',
    #                       milestones_proc=[10],epoch_proc=30,lr_proc=0.01,
    #                       epoch_rec=0)
    model, msg = prune_model(model, train_loader, test_loader, eval_imp=eval_imp_l1norm,
                             input_size=(32, 32), ratio=0.8, ext_type='inner', file_name='c10_res56_l1',
                             milestones_proc=[10], epoch_proc=30, lr_proc=0.01,
                             epoch_rec=0)
