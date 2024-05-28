from script.imps import  *
from data import *

num_cls=100
train_loader = cifar_loader(set='train',num_cls=num_cls, batch_size=128)
test_loader= cifar_loader(set='test',num_cls=num_cls, batch_size=128)
prune_loader = cifar_loader(set='train',num_cls=num_cls, batch_size=512)

model=vggX(num_cls=num_cls)
model.load_state_dict(torch.load('../chk/c10_vgg16'))

# model=resnetC(num_cls=10,num_layer=20)
# model.load_state_dict(torch.load('../chk/c10_res20'))
thres=2
flop_ori, param_ori=calc_flop_para(model,input_size=(32,32),ignore_zero=True)

binding_dicts = ext_vgg(model)
num_dict=len(binding_dicts)
for i in range(num_dict):
    print('dict ', i)
    #################
    binding_dicts = ext_vgg(model)
    # ratio = 0.5
    # prsv_inds=order[(size-int(size*ratio)):]
    # rmv_inds = order[:(size-int(size*ratio))]
    # cmb_inds=[]
    #################
    while True:
        pairs = eval_imp_pcos(model, binding_dicts, dict_ind=i, loader=prune_loader, thres=thres)
        if len(pairs)==0:
            break
        pairs=pairs[:20]
        print(pairs)
        inds1 = [pair['chan_inds'][0] for pair in pairs]
        inds2 = [pair['chan_inds'][1] for pair in pairs]
        chan_opr(binding_dicts[i], inds1=inds1, inds2=inds2,opr_type='fcmbp')

        flop_cut, param_cut = calc_flop_para(model, input_size=(32, 32), ignore_zero=False)
        print('prsev ', flop_cut / flop_ori)
        train_SGD_StepLR(model, train_loader, test_loader, total_epoch=0, lr=0.01, momentum=0.9, weight_decay=5e-4,
                     milestones=[5], gamma=0.1, file_name='', file_dir='../', reg_loss=None, save_process=False)
    #################
    # rmv_inds,cmb_inds=eval_imp_rs(model, binding_dicts, i,prune_loader,rate=1.6)
    # presv_inds=eval_imp_ap(model, binding_dicts, i,prune_loader)
    # if len(presv_inds)==0:
    #     continue
    #################
    #剪枝
    # chan_opr(binding_dicts[i], inds=presv_inds, opr_type='ext')
    # #合并
    # if len(cmb_inds)>0:
    #     print(i,cmb_inds)
    #     inds1 = [pair[0] for pair in cmb_inds]
    #     inds2 = [pair[1] for pair in cmb_inds]
    #     chan_opr(binding_dicts[i], inds1=inds1, inds2=inds2,opr_type='fcmb')
    #格式调整
    # refmt_res(model)
    #################
    # acc = test_model(model, test_loader)
    # print('after acc ', acc)

    #
    # model = copy.deepcopy(model)
    #



#恢复训练
train_SGD_StepLR(model, train_loader, test_loader, total_epoch=20, lr=0.01, momentum=0.9, weight_decay=5e-4,
                    milestones=[5], gamma=0.1, file_name='', file_dir='../', reg_loss=None, save_process=False)
flop_cut, param_cut=calc_flop_para(model,input_size=(32,32))







