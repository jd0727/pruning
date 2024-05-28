from prune_via_gcn.agent import *



num_cls=10
train_loader = cifar3.get_train_loader(num_cls=num_cls, batch_size=128)
test_loader= cifar3.get_test_loader(num_cls=num_cls, batch_size=128)

# 模型
model = CLSframe(backbone='vgg', num_cls=num_cls)
model.load_wei('../chk/c10_vgg16')
binding_dicts = ext_vgg(model._get_backbone())
#统计
flop_ori, para_ori = calc_flop_para(model, input_size=(32,32), ignore_zero=True)
# 打分
scorer = Scorer(model)
# 代理
agent = Agent(scorer=scorer, backbone='gcn')
# agent.load_wei('../chk/ag_vgg')
# 全局代理训练
agent.train_core_muti(model, binding_dicts, train_loader, batch_size=200,
                      total_iter=60, milestones=[20,40], lr_init=0.01,gamma=0.1, show_interval=20)
agent.save_wei('../chk/ag_vgg')



num_dict=len(binding_dicts)
for i in range(num_dict):
    print('Dict ', i)
    #################
    #设置基准
    scorer.set_base(model)
    #层
    binding_dicts = ext_vgg(model._get_backbone())
    binding_dict = binding_dicts[i]#仅使用1层
    #适应性训练
    agent.load_wei('../chk/ag_vgg')
    agent.train_core_muti(model, [binding_dict], train_loader, batch_size=10,
                          total_iter=40, milestones=[20],lr_init=0.01, gamma=0.1, show_interval=20)
    #
    while True:
        binding_dicts = ext_vgg(model._get_backbone())
        binding_dict = binding_dicts[i]
        #确定剪枝
        bx,by=next(iter(train_loader))
        inputs=agent.get_inputs(model,[binding_dict],bx,by)
        masks = agent.get_masks(inputs)
        target_masks = agent.get_target_masks(model, [binding_dict], masks, bx, by, number=50, masks_lib=None)
        mask=target_masks[0]
        #mask=mask.detach().cpu().numpy()
        #
        rmv_inds=np.nonzero(mask==0)[0]
        if len(rmv_inds)==0:
            break
        else:
            rmv_mask=mask[rmv_inds]
            order=np.argsort(rmv_mask)
            rmv_inds=rmv_inds[order]
            rmv_inds=rmv_inds[:20]
        print('Remove',rmv_inds)
        #剪枝
        chan_opr(binding_dict=binding_dict,opr_type='rmv',inds=rmv_inds)
        model=copy.deepcopy(model)
        #统计
        flop_cut, param_cut = calc_flop_para(model, input_size=(32, 32), ignore_zero=True)
        print('prsev ', flop_cut / flop_ori)
        #训练
        if len(rmv_inds)<2:
            break
        train_SGD_StepLR(model, train_loader, test_loader, total_epoch=10, lr=0.001, momentum=0.9, weight_decay=5e-4,
                      milestones=[3], gamma=0.1, file_name='', file_dir='../', reg_loss=None, save_process=False)
    #存档
    model.save_wei('../chk/vgg_save'+str(i))



#恢复训练
train_SGD_StepLR(model, train_loader, test_loader, total_epoch=20, lr=0.01, momentum=0.9, weight_decay=5e-4,
                    milestones=[5], gamma=0.1, file_name='', file_dir='../', reg_loss=None, save_process=False)
flop_cut, param_cut=calc_flop_para(model,input_size=(32,32))







