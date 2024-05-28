from tools import *


#剪枝评价
class Scorer():
    def __init__(self,model,balance=30,retrain=False,input_size=(32,32)):
        self.balance=balance
        self.retrain=retrain
        self.input_size=input_size
        self.set_base(model)

    ############################################################################测试得到适应度
    # 在单层测试 mask 得到loss
    # def mask2loss_onelayer(self,model,binding_dict, mask, bx, by):
    #     #权重缓存
    #     state_dict_old= copy.deepcopy(model.state_dict())
    #     #剪枝
    #     for outer in binding_dict['out']:
    #         outer.weight.data[mask == 0] = 0
    #     #重训练
    #     if self.retrain:
    #         model.train()
    #         optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
    #                                     lr=0.001, momentum=0.9, weight_decay=5e-4)
    #         for i in range(5):
    #             loss=model.get_loss(bx,by)
    #             #
    #             optimizer.zero_grad()
    #             loss.backward()
    #             nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
    #             optimizer.step()
    #     # 测试
    #     model.eval()
    #     with torch.no_grad():
    #         loss = model.get_loss(bx,by)
    #     # 恢复
    #     model.load_state_dict(state_dict_old)
    #     return loss
    def mask2loss_onelayer(self,model,binding_dict, mask, bx, by):
        buffers=[]
        for outer in binding_dict['out']:
            buffer = copy.deepcopy(outer.weight.data)
            buffers.append(buffer)
            outer.weight.data[mask == 0] = 0
        # 测试
        model.eval()
        with torch.no_grad():
            loss = model.get_loss(bx,by)
        # 恢复
        for i,outer in enumerate(binding_dict['out']):
            outer.weight.data = buffers[i]
        return loss
    #计算para数目
    def para_onelayer(self, binding_dict):
        para=0
        for edge in ['in','out','pth']:
            if edge in binding_dict.keys():
                for mod in binding_dict[edge]:
                    para+=mod.weight.data.numel()

        return para

    # 在单层测试mask定义适应度
    def test_mask_onelayer(self,model, binding_dict, mask, bx, by,loss_ori):
        loss = self.mask2loss_onelayer(model,binding_dict, mask, bx, by)
        loss = loss.detach().cpu().numpy()
        #
        loss_incp = (loss) / loss_ori
        para = self.para_onelayer(binding_dict)
        size = binding_dict['size']
        presv_num = np.sum(mask)
        cut_para = (size - presv_num) / size * para
        cut_parap=cut_para/self.para_ori
        #
        # if loss_incp>5:
        #     loss_incp=5+(loss_incp-5)*0.05
        # loss_incp=np.clip(loss_incp,0,100)
        if loss_incp<1:
            loss_incp=loss_incp**3
        else:
            loss_incp=loss_incp**5
        fit = cut_parap*self.balance - loss_incp
        return loss_incp,cut_parap,fit
    ##########################################对外接口
    #设置参考模型base
    def set_base(self,model):
        print('Set Base Model')
        self.model_ori=copy.deepcopy(model)
        flop_ori, para_ori = calc_flop_para(self.model_ori, input_size=self.input_size, ignore_zero=True)
        self.flop_ori=flop_ori
        self.para_ori=para_ori
        return None

    # 在单层测试多个masks 定义适应度
    def test_masks_onelayer(self,model,binding_dict, masks, bx, by):
        self.model_ori.eval()
        with torch.no_grad():
            loss_ori = self.model_ori.get_loss(bx, by)
        loss_ori = loss_ori.detach().cpu().numpy()
        # 测试
        fits = []
        for i, mask in enumerate(masks):
            loss_incp,cut_para,fit=self.test_mask_onelayer(model,binding_dict, mask, bx, by,loss_ori)
            fits.append(fit)
        return fits

    #在多层测试同时剪枝的适应度
    def test_masks_full(self,model,binding_dicts,target_masks, bx, by):
        report=pd.DataFrame(columns=['PruneNum','PrunePrec','LossIncPrec','Fit'])
        self.model_ori.eval()
        with torch.no_grad():
            loss_ori = self.model_ori.get_loss(bx, by)
        loss_ori = loss_ori.detach().cpu().numpy()
        for i, binding_dict in enumerate(binding_dicts):
            target_mask=target_masks[i]
            loss_incp, cut_parap, fit = self.test_mask_onelayer(model,binding_dict, target_mask, bx, by, loss_ori)
            size = binding_dict['size']
            presv_num = np.sum(target_mask)
            #
            report=report.append({
                'PruneNum':size-presv_num,
                'PrunePrec':cut_parap,
                'LossIncPrec':loss_incp,
                'Fit': fit,
            },ignore_index=True)
        fit_ave=np.average(report['Fit'])
        print('-' * 50)
        print(report)
        print('Average Fit:',fit_ave)
        print('-' * 50)
        return report
