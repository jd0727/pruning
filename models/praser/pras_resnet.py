import functools
import torch
from models.resnet_imagenet import resnetI
#直接从res model中提取通道绑定关系

#stage num
def get_stage_num(model):
    stage_num = 4 if hasattr(model, 'stage4') else 3
    return stage_num
#block num
def get_conv_num(block):
    conv_num = 3 if hasattr(block, 'conv3') else 2
    return conv_num


# binding_layers={'size':x,'in':[]...}
#提取res块中间层通道
def ext_inner_chan(model):
    # 得到中间层的dict
    def inner_chan(block, code=1):
        code = int(code)
        convx_name = 'conv' + str(code)
        convx = getattr(block, convx_name)
        channel_size = convx.weight.data.size()[0]
        bn = getattr(block, 'bn' + str(code))
        convn = getattr(block, 'conv' + str(code + 1))
        binding_dict = {
            'size': channel_size,
            'in': [convn],
            'out': [convx],
            'pth': [bn],
            'type':'inner'
        }
        return binding_dict
    #开始提取
    binding_dicts = []
    #得到stage数量
    stage_num = get_stage_num(model)
    for stage_ind in range(1,stage_num+1):
        stage=getattr(model, 'stage'+str(stage_ind))
        # 层内部通道
        for block_name, block in stage.named_children():
            # block内部的channel
            conv_num = get_conv_num(block)
            for conv_ind in range(1,conv_num):
                binding_dicts.append(inner_chan(block, code=conv_ind))
    return binding_dicts

#提取res块加法前通道
def ext_outer_chan(model):
    binding_dicts = []
    stage_num = get_stage_num(model)
    last_conv=model.conv1
    last_bn =model.bn1
    for stage_ind in range(1,stage_num+1):
        stage=getattr(model, 'stage'+str(stage_ind))
        for block_name, block in stage.named_children():
            conv1=getattr(block,'conv1')
            size=conv1.weight.data.size()[1]
            binding_dict = {
                'size': size,
                'in': [conv1],
                'out': [last_conv],
                'pth': [last_bn],
                'type': 'outter'
            }
            binding_dicts.append(binding_dict)
            #连接下一个block
            conv_num = get_conv_num(block)
            last_conv = getattr(block, 'conv' + str(conv_num))
            last_bn = getattr(block, 'bn' + str(conv_num))
    #分类器通道
    size=model.linear.weight.data.size()[1]
    binding_dict = {
        'size': size,
        'in': [model.linear],
        'out': [last_conv],
        'pth': [last_bn],
        'type': 'outter'
    }
    binding_dicts.append(binding_dict)
    return binding_dicts

#提取shortcut通道
def ext_shortcut_chan(model):
    binding_dicts = []
    stage_num = get_stage_num(model)
    for stage_ind in range(1, stage_num + 1):
        stage = getattr(model, 'stage' + str(stage_ind))
        shortcut = getattr(stage[0], 'shortcut')
        if shortcut is not None:

            conv=shortcut[0]
            bn=shortcut[1]
            size=conv.weight.data.size()[0]
            next_conv=getattr(stage[1],'conv1')
            binding_dict = {
                'size': size,
                'in': [next_conv],
                'out': [conv],
                'pth': [bn],
                'type': 'shortcut'
            }
            binding_dicts.append(binding_dict)
    return binding_dicts


#提取res连接通道
def ext_res_chan(model):
    #得到stage数量
    stage_num = get_stage_num(model)
    #逐stage提取
    binding_dicts = []
    for stage_ind in range(1,stage_num+1):
        stage=getattr(model, 'stage'+str(stage_ind))
        #
        convs_out = []
        bn = []
        convs_in = []
        for block_name, block in stage.named_children():
            conv_num =get_conv_num(block)
            # 残差通道输出
            convs_out.append(getattr(block, 'conv' + str(conv_num)))
            bn.append(getattr(block, 'bn' + str(conv_num)))
            # 残差通道输入
            if not block_name == '0':
                convs_in.append(getattr(block, 'conv1'))
        # 确定通道数目
        size = stage[1].conv1.weight.size()[1]  # 第二个block的输入
        # 建立初步dict
        binding_dict_res = {
            'size':size,
            'in': convs_in,
            'out': convs_out,
            'pth': bn,
            'type': 'res'
        }
        # 增加上一层连接
        if stage_ind == 1 and stage[0].shortcut is None:  # 对于res_cifar
            binding_dict_res['out'].insert(0, model.conv1)
            binding_dict_res['pth'].insert(0, model.bn1)
            binding_dict_res['in'].insert(0,stage[0].conv1)
        else:
            binding_dict_res['out'].insert(0,stage[0].shortcut[0])
            binding_dict_res['pth'].insert(0,stage[0].shortcut[1])
        # 增加下一层连接
        if stage_ind == stage_num:
            binding_dict_res['in'].append(model.linear)
        else:
            binding_dict_res['in'].append(getattr(model, 'stage' + str(stage_ind + 1))[0].conv1)
            binding_dict_res['in'].append(getattr(model, 'stage' + str(stage_ind + 1))[0].shortcut[0])
        # 添加binding_layers
        binding_dicts.append(binding_dict_res)

    return binding_dicts

#提取预处理通道
def ext_pre_chan(model):
    binding_dicts=[]
    # 对于res_imagenet还要补上conv1输出通道
    if model.stage1[0].shortcut is not None:
        binding_dict_pic = {
            'size': model.conv1.weight.size()[0],
            'out': [model.conv1],
            'pth': [model.bn1],
            'in': [model.stage1[0].shortcut[0], model.stage1[0].conv1],
            'type': 'pre'
        }
        binding_dicts.append(binding_dict_pic)
    return binding_dicts


#提取res_model所有通道
def ext_resnet(model, meth='all', input_size=None):
    trans_dict={
        'all':['inner','res','pre'],
        'all2':['inner','outer','shortcut']
    }
    func_dict={
        'inner':ext_inner_chan,
        'res': ext_res_chan,
        'pre': ext_pre_chan,
        'outer': ext_outer_chan,
        'shortcut': ext_shortcut_chan
    }
    #转换
    for key in trans_dict.keys():
        if meth==key:
            meth=trans_dict[key]
    #添加函数
    binding_dicts = []
    for name,chan_func in func_dict.items():
        if name in meth:
            binding_dicts += chan_func(model)
    #重排序
    if input_size is not None:
        binding_dicts=sort_dicts(binding_dicts, model, input_size=input_size)
    return binding_dicts

def sort_dicts(binding_dicts, model, input_size=(32,32)):
    order=[]
    def recorder(module, input, output,ind):
        order.append(ind)
        return
    handles=[]
    for i,binding_dict in enumerate(binding_dicts):
        sub_model=binding_dict['out'][0]
        handle = sub_model.register_forward_hook(functools.partial(recorder,ind=i))
        handles.append(handle)
    #规范输入
    if isinstance(input_size,int):
        test_x = torch.rand(size=(5, 3, input_size, input_size))*3
    elif len(input_size)==2:
        test_x = torch.rand(size=(5, 3, input_size[0], input_size[1]))*3
    elif len(input_size)==3:
        test_x = torch.rand(size=(5, input_size[0], input_size[1], input_size[2]))*3
    elif len(input_size)==4:
        test_x = torch.rand(size=input_size)*3
    else:
        print('err size')
        return 0

    device = next(iter(model.parameters())).device
    # device =model.device
    test_x=test_x.to(device)
    model.eval()
    _=model(test_x)
    #重排
    binding_dicts=[binding_dicts[i] for i in order]
    #清除hook
    for hook in handles:
        hook.remove()

    return binding_dicts


if __name__ == '__main__':
    pass
    # model=resnetC(num_layer=20,num_cls=10)
    model = resnetI(num_layer=50, num_cls=10)
    binding_dicts=ext_resnet(model,'all')
    # binding_dicts=sort_dicts(binding_dicts, model, input_size=32)
