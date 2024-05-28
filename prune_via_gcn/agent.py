from prune_via_gcn.gcns import *
from prune_via_gcn.chan_feat_ext import *
from prune_via_gcn.scorer import *
from tools import *
from data import *
from models import *



#剪枝代理
class Agent(nn.Module):
    def __init__(self, scorer,backbone=None, device=None,feat_ext='bn',chans=14,edge_thres=0.25):
        super(Agent, self).__init__()
        #feat_ext
        if feat_ext=='bn':
            self.feat_ext = ext_feats_bn
        elif feat_ext=='lin':
            self.feat_ext = ext_feats_map
        else:
            self.feat_ext= feat_ext
        #scorer
        self.scorer=scorer
        #core
        if backbone == 'gcn':
            self.backbone = GCND(in_channels=chans,out_channels=1)
        elif backbone== 'gcn2':
            self.backbone = GCND2(in_channels=chans,out_channels=1)
        elif backbone== 'gcns':
            self.backbone = GCNS(in_channels=chans,out_channels=1)
        elif backbone== 'mlp':
            self.backbone = GCNDL(in_channels=chans,out_channels=1)
        else:
            self.backbone=backbone
        #device
        if device is None:
            if torch.cuda.is_available():
                inds=ditri_gpu(min_thres=0.1,one_thres=0.5)
                self.device=torch.device('cuda:'+str(inds[0]))
                print('Agent: Put models on cuda ' + str(inds[0]))
                print('Use GPU:',inds)
            else:
                self.device=torch.device('cpu')
        else:
            self.device=device
        self.backbone.to(self.device)
        #cert
        self.cert=nn.BCELoss(reduction='mean')
        self.edge_thres=edge_thres

    ############################################################################从bd中提取特征作为图输入
    # 得到输入图
    def get_inputs(self, model, binding_dicts, bx, by):
        feats = self.feat_ext(model, binding_dicts, bx, by)
        inputs = []
        for i, binding_dict in enumerate(binding_dicts):
            feat=feats[i]
            #
            dist_mat=get_dist_mat(feat,meth='kl')
            # val = np.mean(dist_mat)
            es=get_edges_thres(dist_mat,thres= self.edge_thres)
            es=np.array(es)
            # print('Edg',es.shape[0],'Chan',feat.shape[0])
            es=es.transpose(1,0)
            es=torch.as_tensor(es)
            es = es.to(self.device).long()
            #
            x=torch.as_tensor(feat).float()
            x=x.to(self.device)
            inputs.append((x,es))
        return inputs

    ############################################################################得到合适的target_masks
    #依概率随机采样
    def samp_from_mask(self, mask, number=10):
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()
        #随机
        mask_samps = np.random.rand(number, len(mask))
        for k in range(len(mask)):
            col = mask_samps[:, k]
            thres = 1 - mask[k]
            mask_samps[col > thres, k] = 1
            mask_samps[col < thres, k] = 0
        # 拆分
        mask_samps = [mask_samps[i] for i in range(mask_samps.shape[0])]
        return mask_samps
    #合并同样的masks
    def gether_masks(self, mask_samps):
        number=len(mask_samps)
        #筛选保留
        presv_inds=np.ones(number)
        for i in range(number):
            if presv_inds[i]==0:
                continue
            for j in range(i+1,number):
                if np.all(mask_samps[i] == mask_samps[j]):
                    presv_inds[j]=0
        mask_samps = [mask_samps[i] for i in range(number) if presv_inds[i]==1]
        return mask_samps

    #得到合适的target
    def get_target_masks(self, model,binding_dicts, masks, bx, by, number=20, masks_lib=None):
        target_masks=[]
        for i, binding_dict in enumerate(binding_dicts):
            mask=masks[i]
            mask_samps = self.samp_from_mask(mask, number=number)
            #库中的mask
            if masks_lib is not None:
                mask_lib=masks_lib[i]
                mask_samps+=mask_lib
            #合并同类
            mask_samps=self.gether_masks(mask_samps)
            #适应度
            fits = self.scorer.test_masks_onelayer(model,binding_dict, mask_samps, bx, by)
            #
            ind = np.argmax(fits)
            target_mask = mask_samps[ind]
            target_masks.append(target_mask)
        return target_masks
    ############################################################################应用并训练core

    #应用core得到mask
    def get_masks(self, inputs):
        self.backbone.eval()
        masks = []
        for i, (x,es) in enumerate(inputs):
            mask = self.backbone(x, es)
            mask = torch.squeeze(mask,dim=1)
            mask = mask.to(self.device)
            #
            masks.append(mask)
        return masks

    #利用target_masks单次随机训练
    def train_batch(self, inputs, target_masks, optimzer,num_iter=200,balance=False):
        self.backbone.train()
        for i in range(num_iter):
            #随机指针
            ptr=int(np.random.randint(0, len(inputs)))
            x, es=inputs[ptr]
            target=target_masks[ptr]
            if isinstance(target,np.ndarray):
                target=torch.as_tensor(target).float()
                target=target.to(self.device)
            #训练
            pred=self.backbone(x, es)
            pred = torch.squeeze(pred)
            # 自适应平衡样本
            if balance:
                # gamma=2
                # inds_pos = torch.squeeze(torch.nonzero(target == 1),dim=1)
                # inds_neg = torch.squeeze(torch.nonzero(target == 0),dim=1)
                # loss_pos =-torch.pow(1-pred[inds_pos],gamma)*torch.log(pred[inds_pos]+1e-7)
                # loss_neg =-torch.pow(pred[inds_neg],gamma)*torch.log(1-pred[inds_neg]+1e-7)
                # loss=(torch.sum(loss_pos)+torch.sum(loss_neg))/target.size(0)
                inds_pos = torch.squeeze(torch.nonzero(target == 1),dim=1)
                inds_neg = torch.squeeze(torch.nonzero(target == 0),dim=1)

                inds_pos=torch.randperm(inds_pos.size(0))
                inds_neg = torch.randperm(inds_neg.size(0))
                inds_pos = inds_pos[:inds_neg.size(0)]
                inds_neg = inds_neg[:inds_pos.size(0)]

                loss_pos =-torch.log(pred[inds_pos]+1e-7)
                loss_neg =-torch.log(1-pred[inds_neg]+1e-7)
                loss=(torch.sum(loss_pos)+torch.sum(loss_neg))/(inds_neg.size(0)+inds_pos.size(0))
            else:
                loss=self.cert(pred,target)
            #梯度
            optimzer.zero_grad()
            loss.backward()
            optimzer.step()
        return None

    #利用数据集多次训练
    def train_core_muti(self, model, binding_dicts, train_loader, batch_size=200,
                        total_iter=100, milestones=None,lr_init=0.1, gamma=0.1, show_interval=20,
                        samp_num=30,lib_size=20,record=False):
        if milestones is None:
            milestones = [40, 50]
        #优化器
        print('Start train')
        optimzer=torch.optim.SGD(filter(lambda p: p.requires_grad, self.backbone.parameters()),
                                      lr=lr_init, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimzer, milestones=milestones, gamma=gamma)
        #建立库
        masks_lib = []
        for _ in range(len(binding_dicts)):
            masks_lib.append([])
        data=pd.DataFrame(columns=['mask','tar_mask','enp','iter'])
        #开始训练
        n=1
        while True:
            for _, (bx, by) in enumerate(train_loader):
                inputs = self.get_inputs(model.backbone, binding_dicts, bx, by)
                masks = self.get_masks(inputs)
                # 测试剪枝
                target_masks = self.get_target_masks(model,binding_dicts, masks, bx, by, number=samp_num,masks_lib=masks_lib)
                #添加库
                if lib_size>0:
                    for k, tm in enumerate(target_masks):
                        if isinstance(tm, torch.Tensor):
                            tm = tm.detach().cpu().numpy()
                        masks_lib[k].append(tm)
                        if len(masks_lib[k]) > lib_size:  # 达到容量
                            del (masks_lib[k][0])

                # 训练
                # balance=n>150
                # balance = np.random.rand()>0.8
                balance=False
                self.train_batch(inputs, target_masks,optimzer, num_iter=batch_size,balance=balance)
                scheduler.step()
                #计算参数
                masks=[m.detach().cpu().numpy() for m in masks]
                enps=[np.mean(-np.log(m+1e-7)*m) for m in masks]
                enp=np.mean(enps)
                #展示
                if n==1 or n%show_interval==0 or n == total_iter:
                    print('Iter', n,' Enp',enp)
                    np.set_printoptions(precision=3, suppress=False)
                    # print('TarMask[0]', target_masks[0])
                    print('Chans[0]', inputs[0][0].size(0),'Edgs[0]',inputs[0][1].size(1))
                    print('Mask[0]', masks[0])
                    self.scorer.test_masks_full(model,binding_dicts, target_masks, bx, by)
                    if record:
                        data=data.append({
                            'mask':masks[0],
                            'tar_mask':target_masks[0],
                            'enp':enp,
                            'iter':n
                        },ignore_index=True)

                #收敛截止条件
                masks_nums=[len(self.gether_masks(m)) for m in masks_lib]
                if enp<0.1 and all(np.array(masks_nums)==1):
                    print('Stable End')
                    n=total_iter
                    break
                #轮次截止条件
                if n == total_iter:
                    print('Number End')
                    break
                else:
                    n=n+1
            if n == total_iter:
                break

        return target_masks
    ############################################################################权重存取
    def _get_backbone(self):
        if 'cuda' in self.device.type and isinstance(self.backbone,nn.DataParallel):
            backbone = self.backbone.module
        else:
            backbone = self.backbone
        return backbone

    def save_wei(self, filename):
        backbone = self._get_backbone()
        torch.save(backbone.state_dict(), filename)

    def load_wei(self, filename):
        backbone = self._get_backbone()
        state_dict = torch.load(filename, map_location=self.device)
        backbone.load_state_dict(state_dict)

    def load_wei_fmt(self, filename='../../chk/on_voc'):
        backbone = self._get_backbone()
        load_fmt(backbone, filename)
    ############################################################################评估剪枝计划
    @staticmethod
    def eval_plan(binding_dicts):
        plan=pd.DataFrame(columns=['Recvrs','Pths','Chans'])
        for i,binding_dict in enumerate(binding_dicts):
            recvrs=len(binding_dict['in'])
            if 'pth' in binding_dict.keys():
                pths=len(binding_dict['pth'])
            else:
                pths=0
            chans=12*recvrs
            plan=plan.append({
                'Recvrs':recvrs,
                'Pths':pths,
                'Chans':chans
            },ignore_index=True)
        print(plan)
        return plan

if __name__ == '__main__':
    num_cls=10
    test_loader= cifar_loader(set='test',num_cls=num_cls, batch_size=128)
    train_loader= cifar_loader(set='train',num_cls=num_cls, batch_size=128)
    #模型
    model=CLSframe(backbone='vgg',num_cls=num_cls)
    model.load_wei('../chk/c10_vgg16')
    binding_dicts = ext_vgg(model._get_backbone())
    #打分
    scorer=Scorer(model)
    #代理
    agent = Agent(scorer=scorer,backbone='gcn')
    #训练
    agent.load_wei('../chk/ag_vgg')
    agent.train_core_muti(model, binding_dicts, train_loader, batch_size=200,
                          total_iter=20, milestones=[10], gamma=0.01, show_interval=10)

    agent.save_wei('../chk/ag_vgg')

    # for i, (bx, by) in enumerate(train_loader):
    #
    # # (batch_x,batch_y)=next(iter(test_loader))
    #     inputs=pruner.get_inputs(model.backbone, binding_dicts, bx, by)
    #     masks=pruner.get_masks(inputs)
    #     #测试剪枝
    #     target_masks=pruner.get_target_masks(model, binding_dicts, masks, bx, by, number=20)
    #     #训练
    #     pruner.train_core_once(inputs, target_masks, num_iter=200)
    #     pruner.test_core(model, binding_dicts, target_masks, bx, by)
    #     print(target_masks[0])
    #     if i%10==0:
    #         print(masks[0].detach().cpu().numpy())


    



