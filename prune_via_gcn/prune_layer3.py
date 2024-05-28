from prune_via_gcn.agent import *


#
num_cls=100
input_size=(32,32)
train_loader = cifar_loader(set='train',num_cls=num_cls, batch_size=128)
test_loader= cifar_loader(set='test',num_cls=num_cls, batch_size=128)
prune_loader = cifar_loader(set='train',num_cls=num_cls, batch_size=512)


# num_cls=200
# input_size=(64,64)
# test_loader= timg_loader(set='test', batch_size=64)
# train_loader= timg_loader(set='train', batch_size=64)
# prune_loader = timg_loader(set='train', batch_size=256)

bal=100
for edge_thres in [0]:
    # 模型
    model = CLSframe(backbone='vgg', num_cls=num_cls,drop_rate=0.0)
    #model = CLSframe(backbone='resC', num_cls=num_cls,num_layer=20)
    # model = CLSframe(backbone='resI', num_cls=num_cls,num_layer=50)

    model.load_wei('../chk/c100_vgg16')
    #model.load_wei('../chk/img_vgg16_px_n')
    # model.load_wei('../chk/save21')
    # model.load_wei('../chk/save35')
    # model.load_wei('../chk/img_res50')

    binding_dicts = model.get_dicts(meth='all')

    #统计
    flop_ori, para_ori = calc_flop_para(model, input_size=input_size, ignore_zero=True)

    model2 =  CLSframe(backbone='vgg', num_cls=num_cls,drop_rate=0.0)
    # model2 =  CLSframe(backbone='resI', num_cls=num_cls,num_layer=50)
    #model2 = CLSframe(backbone='resC', num_cls=num_cls,num_layer=20)

    # model2.load_wei('../chk/img_res50')
    # model2.load_wei('../chk/c10_res56_p/c10_res56_p54')
    model2.load_wei('../chk/c100_vgg16')


    scorer = Scorer(model2,balance=bal,retrain=False,input_size=input_size)

    print('Get Plan')
    plan=Agent.eval_plan(binding_dicts)
    #
    num_dict=len(binding_dicts)
    for i in range(0,num_dict):
        print('Dict ', i)
        #################
        #层
        binding_dicts =model.get_dicts(meth='all')
        binding_dict = binding_dicts[i]#仅使用1层
        #考虑跳过
        if binding_dict['size']<=1:
            continue
        agent = Agent(scorer=scorer, backbone='gcn', feat_ext= 'lin', chans= plan['Chans'][i] ,edge_thres=edge_thres)

        target_masks=agent.train_core_muti(model, [binding_dict], prune_loader, batch_size=5,
                              total_iter=300, milestones=[50,100],lr_init=0.01, gamma=0.1, show_interval=40,
                              samp_num=30,lib_size=20)
        mask=target_masks[0]
        #
        rmv_inds=np.nonzero(mask==0)[0]
        if len(rmv_inds)==0:
            print('Remove None')
            continue
        elif len(rmv_inds)==binding_dict['size']:
            rmv_inds=rmv_inds[1:]
        print('Remove Num',len(rmv_inds),'Inds',rmv_inds)
        #剪枝
        chan_opr(binding_dict=binding_dict,opr_type='rmv',inds=rmv_inds)
        model=copy.deepcopy(model)
        #统计
        flop_cut, param_cut = calc_flop_para(model, input_size=input_size, ignore_zero=True)
        print('Prsev', flop_cut / flop_ori)

        train_SGD_StepLR(model, train_loader, test_loader, total_epoch=30, lr=0.0001, momentum=0.9, weight_decay=5e-4,
                      milestones=[7], gamma=0.1, file_name='',reg_loss=None, save_process=False)
        #存档
        model.save_wei('../chk/save'+str(i))

    model.save_wei('../chk/c10_vgg16_b100_t'+str(edge_thres))








