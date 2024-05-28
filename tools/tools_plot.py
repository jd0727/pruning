import matplotlib.pyplot as plt
plt.rc('font',family='Times New Roman')
import numpy as np
import matplotlib
# matplotlib.use('TkAgg')
import pandas as pd
import math
import pickle


#####################################################################排版
#按数列展示
def porder_arr(datas, shower=None, **kwargs):
    #确定长宽
    num_map = datas.shape[0]
    area = 10 * 8
    num_wid = math.ceil(np.sqrt(num_map))
    num_hei = math.ceil(num_map / num_wid)
    rate = num_wid / num_hei
    wid = round(math.sqrt(area / rate))
    hei = round(area / wid)
    fig = plt.figure(figsize=(wid, hei))
    # 画图
    ind = 1
    for i in range(num_wid):
        for j in range(num_hei):
            if ind - 1 == num_map:
                break
            ax = fig.add_subplot(num_wid, num_hei, ind)
            ax.set_title(str(ind - 1), fontdict={
                'family': 'DejaVu Sans',
                'weight': 'bold',
                'size': min(25, 12 * 6 / num_hei),
            })
            shower(axis=ax, data=datas[ind - 1], **kwargs)
            ind += 1
    fig.subplots_adjust(wspace=0.3, hspace=0.6)
    return fig

#按矩阵展示
def porder_mat(datas, shower=None, **kwargs):
    shape=datas.shape
    rate=shape[0]/shape[1]
    area=10*8
    wid=round(math.sqrt(area/rate))
    hei=round(area/wid)
    fig=plt.figure(figsize=(wid,hei))
    #设置刻度
    ax=fig.add_subplot(1,1,1)
    #x
    ax.set_xlim(0,shape[1])
    ax.set_xticks(np.arange(shape[1])+0.5)
    ax.set_xticklabels([str(i) for i in range(shape[1])],fontdict={
        'family': 'DejaVu Sans',
        'weight': 'bold',
        'size': min(25,25*12/shape[1]),
         })
    ax.xaxis.tick_top()
    #y
    ax.set_ylim(0,shape[0])
    ax.set_yticks(np.arange(shape[0])+0.5)
    ax.set_yticklabels([str(i) for i in range(shape[0])],fontdict={
        'family': 'DejaVu Sans',
        'weight': 'bold',
        'size': min(25,25*12/shape[0]),
         })
    ax.invert_yaxis()
    #设置边框
    ax.spines['bottom'].set_linewidth(0)
    ax.spines['left'].set_linewidth(0)
    ax.spines['right'].set_linewidth(0)
    ax.spines['top'].set_linewidth(0)
    ax.tick_params(axis=u'both', which=u'both',length=0)
    # 画图
    ind = 1
    for i in range(shape[0]):
        for j in range(shape[1]):
            ax = fig.add_subplot(shape[0], shape[1], ind)
            shower(axis=ax, data=datas[i][j], **kwargs)
            ind += 1
    # fig.subplots_adjust(wspace=0.3, hspace=0.6)
    return fig
#####################################################################展示
#检查数据分布
def show_distribute(data, quant_step=None, num_quant=20, axis=None):
    std=np.std(data)
    mean=np.average(data)
    if quant_step==None:
        quant_step=std/5
    #检查是否有值
    if std==0:
        vals_axis=[mean-1,mean,mean+1]
        nums=[0,len(data),0]
        quant_step=0.1
    else:
        vals_axis=(np.arange(-num_quant,num_quant)+0.5)*quant_step+mean
        nums=np.zeros(shape=num_quant*2)
        for i in range(len(data)):
            ind=np.floor((data[i]-mean)/quant_step)+num_quant
            if ind<num_quant*2:
                nums[int(ind)]+=1
        #归一化
        nums=nums/np.sum(nums)/quant_step
    #画图
    if axis is None:
        fig, axis = plt.subplots()
    axis.bar(vals_axis, nums, width=quant_step * 0.8, color='k')
    return vals_axis,nums
#图片展示
def show_img(data, axis=None, **kwargs):
    if axis is None:
        fig, axis = plt.subplots()
    axis.imshow(data, cmap=plt.get_cmap('Greys'), **kwargs)
    axis.axis('off')
    return None
#显示曲线
def show_curve(data, axis=None, **kwargs):
    if axis is None:
        fig, axis = plt.subplots()
    axis.plot(data, color='k',**kwargs)
    return None

#显示分布
def show_distri(means,vars, axis=None, **kwargs):
    if axis is None:
        fig, axis = plt.subplots()
    assert len(means)==len(vars),'len err'
    wid=0.2
    for i in range(len(means)):
        color='k'
        axis.plot([i, i], [means[i]-vars[i], means[i]+vars[i]],color=color)
        axis.plot([i-wid, i+wid], [means[i] + vars[i], means[i] + vars[i]],color=color)
        axis.plot([i-wid, i+wid], [means[i] - vars[i], means[i] - vars[i]],color=color)
        axis.plot(i, means[i], color='r', marker='x', markersize=20)
    return None

#####################################################################
#显示fliters
def show_fliters(fliters):
    assert len(fliters.shape) == 4, 'shape need be 4'
    # 设置归一化
    vmin = np.min(fliters)
    vmax = np.max(fliters)
    fig=porder_mat(fliters, shower=show_img, vmin=vmin, vmax=vmax)
    return fig
#显示maps
def show_maps(feat_maps):
    assert len(feat_maps.shape) == 3, 'shape need be 3'
    # 设置归一化
    vmin = np.min(feat_maps)
    vmax = np.max(feat_maps)
    fig=porder_arr(feat_maps, shower=show_img, vmin=vmin, vmax=vmax)
    return fig

#####################################################################

if __name__ == '__main__':
    maps = np.random.rand(12,1)
    plt.imshow(maps)
    plt.colorbar()
    plt.show()

    # show_maps(feat_maps)
    # porder_mat(feat_maps, shower=show_img)
