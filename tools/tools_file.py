import numpy as np
import pickle
import copy
import random
import types
import os
import pandas as pd
#文件操作
##############################################################路径设置
save_dir='../save'
xls_dir='../xls'
##############################################################对象存取
#保存对象
def save(obj,file_name):
    with open(file_name, 'wb+') as f:
        pickle.dump(obj, f)
    return None
#读取对象
def load(file_name):
    with open(file_name, 'rb+') as f:
        obj=pickle.load(f)
    return obj
##############################################################工作区存取
#保存当前工作区
ignore_names=['__name__','__doc__','__package__','__loader__','__spec__','__file__','__builtins__']
def save_space(locals,file_name,dir_name=save_dir):
    save_dict={}
    for name,val in locals.items():
        if callable(val):
            continue
        if name in ignore_names:
            continue
        if type(val)==types.ModuleType:
            continue
        print('saving',name,type(val))
        save_dict[name]=copy.copy(val)
    #保存
    save(save_dict,file_name,dir_name)
    return None

#恢复工作区
def load_space(locals,file_name,dir_name=save_dir):
    save_dict=load(file_name,dir_name)
    for name,val in save_dict.items():
        locals[name]=val
#####################################################################xlsx操作
#写xls
def numpy2xlsx(data,file_name,dir_name=xls_dir):
    if not file_name.endswith('.xlsx'):
        file_name+='.xlsx'
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    file_path = os.path.join(dir_name, file_name)
    df = pd.DataFrame(data,columns=None,index=None)
    df.to_excel(file_path, index=False,header=False)
    return
#读xls
def xlsx2numpy(file_name,dir_name=xls_dir):
    if not file_name.endswith('.xlsx'):
        file_name+='.xlsx'
    file_path = os.path.join(dir_name, file_name)
    data=pd.read_excel(file_path,header=None)
    data = np.array(data)
    return data
