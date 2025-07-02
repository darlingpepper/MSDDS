import pandas as pd
import numpy as np
import json
import csv
import os
from sklearn.model_selection import KFold, train_test_split
from rdkit.Chem import rdMolDescriptors
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit import Chem
from rdkit.Chem import AllChem
import os

import shutil
import cv2
import imghdr
from PIL import Image
import shutil
from PIL import ImageFile
import timm
import torch
import random
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import numpy as np
from torchvision import transforms

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='drugcomb', help='Name of the dataset (e.g., drugcomb, drugcombdb)')
 # 解析命令行参数
args = parser.parse_args()
data = args.data
#构建cell line
cell_list = pd.read_csv(f'{data}/{data}_cellid.csv')['DepMap ID'].values
gene_list = pd.read_csv('./cell/gene_4079.csv')['entrez_id'].values
print(len(cell_list))

# 'tanh_norm'
def normalization(data): 
    std1 = np.nanstd(data, axis=0)  
    data = np.ascontiguousarray(data)
    means1 = np.mean(data, axis=0) 
    data = (data-means1)/std1
    data = np.tanh(data)
    data[np.isnan(data)] = 0
    return data

def process_gene(df):
    for i in gene_list.astype('str'):
        if i not in df.columns:
            df[i] = 0.
    mean_arr = df.mean()
    for j in cell_list:
        if j not in df.index:
            df.loc[j,:] = mean_arr
    df = df[gene_list.astype('str')]
    df = df.loc[cell_list,:]
    df_arr = normalization(df.values)
    print(df_arr.shape)
    return df_arr

exp_raw = pd.read_csv('./cell/cell_exp_raw.csv', index_col=0)
mut_raw = pd.read_csv('./cell/cell_mut_raw.csv', index_col=0)
cn_raw = pd.read_csv('./cell/cell_cn_raw.csv', index_col=0)
eff_raw = pd.read_csv('./cell/cell_eff_raw.csv', index_col=0)
dep_raw = pd.read_csv('./cell/cell_dep_raw.csv', index_col=0)
met_raw = pd.read_csv('./cell/cell_met_raw.csv', index_col=0)


print(exp_raw.shape)
print(mut_raw.shape)
print(cn_raw.shape)
print(eff_raw.shape)
print(dep_raw.shape)
print(met_raw.shape)
exp_norm = process_gene(exp_raw)
print(1)
mut_norm = process_gene(mut_raw)
print(2)
cn_norm = process_gene(cn_raw)
print(3)
eff_norm = process_gene(eff_raw)
print(4)
dep_norm = process_gene(dep_raw)
print(5)
met_norm = process_gene(met_raw)
print(6)

# stack six-omics
exp_mut_cn_eff_dep_met = np.dstack((exp_norm,mut_norm,cn_norm,eff_norm,dep_norm,met_norm))
exp_mut_cn_eff_dep_met_dict = dict(zip(cell_list,exp_mut_cn_eff_dep_met))
print(len(exp_mut_cn_eff_dep_met_dict))

np.save(f'./{data}/cell.npy',exp_mut_cn_eff_dep_met_dict)



#3d-img
def generate_3d_comformer(smiles, sdf_save_path, mmffVariant="MMFF94", randomSeed=0, maxIters=5000, increment=2, optim_count=10, save_force=False):
    count = 0
    while count < optim_count:
        m = Chem.MolFromSmiles(smiles)
        m3d = Chem.AddHs(m)
        if save_force:
            try:
                AllChem.EmbedMolecule(m3d, randomSeed=randomSeed)
                res = AllChem.MMFFOptimizeMolecule(m3d, mmffVariant=mmffVariant, maxIters=maxIters)
                m3d = Chem.RemoveHs(m3d)
            except:
                m3d = Chem.RemoveHs(m3d)
                print("forcing saving molecule which can't be optimized ...")
                break
        else:
            AllChem.EmbedMolecule(m3d, randomSeed=randomSeed)
            res = AllChem.MMFFOptimizeMolecule(m3d, mmffVariant=mmffVariant, maxIters=maxIters)
            print(res)
            m3d = Chem.RemoveHs(m3d)
        if res == 1:
            maxIters = maxIters * increment
            count += 1
            continue
        Chem.MolToMolFile(m3d, sdf_save_path)
        break
    if save_force:
        print("forcing saving molecule without convergence ...")
        Chem.MolToMolFile(m3d, sdf_save_path)


drugid_smiles_df = pd.read_csv(f'{data}/drugid_smiles.csv')
drug_dict = dict(zip(drugid_smiles_df.iloc[:, 0], drugid_smiles_df.iloc[:, 1]))
if os.path.exists(f'{data}/sdf'):
    shutil.rmtree(f'{data}/sdf')
os.makedirs(f'{data}/sdf')
for name, smiles in drug_dict.items():
    sdf_save_path = f"{data}/sdf/{name}.sdf"  # 为每个分子生成一个SDF文件
    print(f"Processing SMILES for {name}: {smiles}")
    generate_3d_comformer(smiles, sdf_save_path, save_force=True)

print("sdf done")


import pymol
from pymol import cmd
import __main__
__main__.pymol_argv = ['pymol', '-qc']  # '-q' 表示静默模式，'-c' 表示无 GUI
# 定义参数
input_folder = f"{data}/sdf"  # 包含 SDF 文件的输入文件夹
output_folder = f"{data}/sdf_img"  # 保存图像的输出文件夹
rotate_direction = "x"  # 旋转方向：x, y, z
rotate_angle = 30  # 旋转角度
image_format = "png"  # 输出图像格式


if os.path.exists(output_folder): 
    shutil.rmtree(output_folder)
os.mkdir(output_folder)

# 初始化 PyMOL
pymol.finish_launching()
# 遍历输入文件夹中的所有 SDF 文件
for sdf_file in os.listdir(input_folder):
    if sdf_file.endswith(".sdf"):
        sdf_path = os.path.join(input_folder, sdf_file)
        molecule_name = os.path.splitext(sdf_file)[0]  # 获取分子名称
        output_image_folder = f"{output_folder}/{molecule_name}"
        if os.path.exists(output_image_folder): 
            shutil.rmtree(output_image_folder)
        os.mkdir(output_image_folder)
        # output_image_path = os.path.join(output_folder, f"{molecule_name}.{image_format}")
        # 加载分子
        cmd.load(sdf_path, molecule_name)
        # 设置背景颜色
        cmd.bg_color("white")
        # 隐藏氢原子
        cmd.hide("(hydro)")
        # 设置显示样式
        cmd.set("stick_ball", 1)
        cmd.set("stick_ball_ratio", 3.5)
        cmd.set("stick_radius", 0.15)
        cmd.set("sphere_scale", 0.2)
        cmd.set("valence", 1)
        cmd.set("valence_mode", 0)
        cmd.set("valence_size", 0.1)
        # 保存图像0
        output_image_path = f"{output_image_folder}/0.png"
        cmd.save(output_image_path)
        # 旋转分子
        cmd.rotate("x", 90, molecule_name)
        # 保存图像1
        output_image_path = f"{output_image_folder}/1.png"
        cmd.save(output_image_path)
        # 旋转分子
        cmd.rotate("x", 90, molecule_name)
        # 保存图像2
        output_image_path = f"{output_image_folder}/2.png"
        cmd.save(output_image_path)
        # 旋转分子
        cmd.rotate("x", 90, molecule_name)
        # 保存图像3
        output_image_path = f"{output_image_folder}/3.png"
        cmd.save(output_image_path)
        # 旋转分子
        cmd.rotate("x", 90, molecule_name)
        cmd.rotate("y", 90, molecule_name)
        # 保存图像4
        output_image_path = f"{output_image_folder}/4.png"
        cmd.save(output_image_path)
        # 旋转分子
        cmd.rotate("y", 180, molecule_name)
        # 保存图像5
        output_image_path = f"{output_image_folder}/5.png"
        cmd.save(output_image_path)
        # 删除当前加载的分子
        cmd.delete(molecule_name)

# 退出 PyMOL
pymol.cmd.quit()


def padding_white_and_resize(img_path, trt_path, new_h, new_w, resize_h, resize_w):
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    if imghdr.what(img_path) == "png":
        img = Image.open(img_path).convert("RGB")
        img.save(img_path)
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"无法加载图像：{img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, c = img.shape
    assert (new_w - w) % 2 == 0 and (new_h - h) % 2 == 0, "新的宽高必须能容纳原始图像并居中"
    # 使用cv2.copyMakeBorder填充白色背景
    border = ((new_h - h) // 2, (new_w - w) // 2)
    new_img = cv2.copyMakeBorder(img, border[0], border[0], border[1], border[1], cv2.BORDER_CONSTANT, value=[255, 255, 255])
    # 调整大小并保存
    new_img = cv2.resize(new_img, (resize_h, resize_w), interpolation=cv2.INTER_AREA)
    cv2.imwrite(trt_path, cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR))
    print(f"处理完成，结果保存到：{trt_path}")


input_folder = f"{data}/sdf_img"
output_folder = f"{data}/3d_img"
if os.path.exists(output_folder): 
    shutil.rmtree(output_folder)
os.mkdir(output_folder)

for folder_path, subfolders, files in os.walk(input_folder):
    print(folder_path)
    drugid = folder_path.split("/")[-1]  # 按照 '/' 分割字符串
    print(drugid)
    output_folder_sub = f'{output_folder}/{drugid}'
    if os.path.exists(output_folder_sub): 
        shutil.rmtree(output_folder_sub)
    os.mkdir(output_folder_sub)
    for img in files:
        img_path = f"{folder_path}/{img}"
        trt_path = f"{output_folder_sub}/{img}"
        # 示例运行
        padding_white_and_resize(img_path=img_path,trt_path=trt_path,new_h=640,new_w=640,resize_h=224, resize_w=224)
print("3d_img done")

class FeatureExtractionModule(torch.nn.Module):
    def __init__(self, head_arch, num_tasks, head_arch_params=None, **kwargs):
        super(FeatureExtractionModule, self).__init__()

        self.head_arch = head_arch
        self.num_tasks = num_tasks
        if head_arch_params is None:
            head_arch_params = {"inner_dim": None, "dropout": 0.2, "activation_fn": None}
        self.head_arch_params = head_arch_params
        self.model = timm.create_model("vit_small_patch16_224", pretrained=False, **kwargs)

    def forward(self, x):
        return self.model.forward_features(x)[:, 0, :]

def load_pretrained_component(model, pretrained_pth, model_key, consistency=True):
    flag = False  # load successfully when only flag is true
    desc = None
    if pretrained_pth:
        if os.path.isfile(pretrained_pth):
            print("===> Loading checkpoint '{}'".format(pretrained_pth))
            checkpoint = torch.load(pretrained_pth, map_location='cpu')
            # load parameters
            ckpt_model_state_dict = checkpoint[model_key]
            if consistency:  # model and ckpt_model_state_dict is consistent.
                model.load_state_dict(ckpt_model_state_dict)
                print("load all the parameters of pre-trianed model.")
            else:  # load parameter of layer-wise, resnet18 should load 120 layer at head.
                ckp_keys = list(ckpt_model_state_dict)
                cur_keys = list(model.state_dict())
                len_ckp_keys = len(ckp_keys)
                len_cur_keys = len(cur_keys)
                model_sd = model.state_dict()
                for idx in range(min(len_ckp_keys, len_cur_keys)):
                    ckp_key, cur_key = ckp_keys[idx], cur_keys[idx]
                    model_sd[cur_key] = ckpt_model_state_dict[ckp_key]
                model.load_state_dict(model_sd)
                print("load the first {} parameters. layer number: model({}), pretrain({})"
                         .format(min(len_ckp_keys, len_cur_keys), len_cur_keys, len_ckp_keys))
            desc = "[resume model info] The pretrained_model is at checkpoint {}. \t info: {}"\
                .format(checkpoint['epoch'], checkpoint['desc'])
            flag = True
        else:
            print("===> No checkpoint found at '{}'".format(pretrained_pth))
    else:
        print('===> No pre-trained model')
    return flag, desc


# img_path = f"{data}/sdf_img/144/0.png"
mean, std = IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
img_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
videoMol = FeatureExtractionModule(head_arch="arch1", num_tasks=5)
load_flag, resume_desc = load_pretrained_component(videoMol, 'ckpts/videomol/VideoMol_vit_small_patch16_224.pth', "videomol", consistency=False)
print(123)
print(load_flag, resume_desc)
print(123)


label_df = pd.read_csv(f'{data}/drugid_smiles.csv')
drug_id = label_df['id']
# 初始化一个字典来存储结果
results_dict = {}
print("start:")
for id in drug_id:
    img_path = f'{data}/3d_img/{id}'
    img_0_path = f'{img_path}/0.png'
    img_1_path = f'{img_path}/1.png'
    img_2_path = f'{img_path}/2.png'
    img_3_path = f'{img_path}/3.png'
    img_4_path = f'{img_path}/4.png'
    img_5_path = f'{img_path}/5.png'
    img_0 = Image.open(img_0_path).convert("RGB")
    img_1 = Image.open(img_1_path).convert("RGB")
    img_2 = Image.open(img_2_path).convert("RGB")
    img_3 = Image.open(img_3_path).convert("RGB")
    img_4 = Image.open(img_4_path).convert("RGB")
    img_5 = Image.open(img_5_path).convert("RGB")
    img_0 = img_transforms(img_0).unsqueeze(0)
    img_1 = img_transforms(img_1).unsqueeze(0)
    img_2 = img_transforms(img_2).unsqueeze(0)
    img_3 = img_transforms(img_3).unsqueeze(0)
    img_4 = img_transforms(img_4).unsqueeze(0)
    img_5 = img_transforms(img_5).unsqueeze(0)
    x0 = videoMol(img_0)
    x1 = videoMol(img_1)
    x2 = videoMol(img_2)
    x3 = videoMol(img_3)
    x4 = videoMol(img_4)
    x5 = videoMol(img_5)
    array0 = x0.detach().numpy()
    array1 = x1.detach().numpy()
    array2 = x2.detach().numpy()
    array3 = x3.detach().numpy()
    array4 = x4.detach().numpy()
    array5 = x5.detach().numpy()
    # 沿着第一个维度拼接
    concatenated_array = np.concatenate((array0, array1, array2, array3, array4, array5), axis=0)
    # 打印结果
    print("Shape of concatenated array:", concatenated_array.shape)
    # 将结果存储到字典中
    results_dict[id] = concatenated_array

print(results_dict)
np.save(f'{data}/3dvideomolEmbed.npy',results_dict)

print("3d_feature done")


#2d_img
def loadSmilesAndSave(smis, path):
    '''
        smis: e.g. COC1=C(C=CC(=C1)NS(=O)(=O)C)C2=CN=CN3C2=CC=C3
        path: E:/a/b/c.png
        ==============================================================================================================
        demo:
            smiless = ["OC[C@@H](NC(=O)C(Cl)Cl)[C@H](O)C1=CC=C(C=C1)[N+]([O-])=O", "CN1CCN(CC1)C(C1=CC=CC=C1)C1=CC=C(Cl)C=C1",
              "[H][C@@](O)(CO)[C@@]([H])(O)[C@]([H])(O)[C@@]([H])(O)C=O", "CNC(NCCSCC1=CC=C(CN(C)C)O1)=C[N+]([O-])=O",
              "[H]C(=O)[C@H](O)[C@@H](O)[C@@H](O)[C@H](O)CO", "CC[C@H](C)[C@H](NC(=O)[C@H](CC1=CC=C(O)C=C1)NC(=O)[C@@H](NC(=O)[C@H](CCCN=C(N)N)NC(=O)[C@@H](N)CC(O)=O)C(C)C)C(=O)N[C@@H](CC1=CN=CN1)C(=O)N1CCC[C@H]1C(=O)N[C@@H](CC1=CC=CC=C1)C(O)=O"]

            for idx, smiles in enumerate(smiless):
                loadSmilesAndSave(smiles, "{}.png".format(idx+1))
        ==============================================================================================================
    '''
    mol = Chem.MolFromSmiles(smis)
    img = Draw.MolsToGridImage([mol], molsPerRow=1, subImgSize=(224, 224))
    img.save(path)

#smiles转image
def smiles_image(path):
    image_folder_path = f'{path}/2d_img'
    # 检查文件夹是否存在
    if os.path.exists(image_folder_path):
    # 如果文件夹存在，则删除它
        shutil.rmtree(image_folder_path)
    # 创建新的文件夹
    os.makedirs(image_folder_path)
    with open(f'{path}/drugid_smiles.csv', 'r') as file:
        # 创建CSV读取器
        reader = csv.reader(file)
        # 跳过第一行（标题行）
        next(reader)
        for line in reader:
            did, smiles = line[0], line[1]
            path_image = f'{image_folder_path}/{did}.png'
            loadSmilesAndSave(smiles, path_image)
    print("smiles转image完成")

smiles_image(data)

print("2d_img done")

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import torchvision
import torch
import torch.nn as nn
import numpy as np

def load_pretrained_component(pretrained_pth, model_key, consistency=False, logger=None,modelname = "ResNet18"):
    if modelname == "ResNet18":
        model = torchvision.models.resnet18(pretrained=False)
        # model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif modelname == "ResNet34":
        model = torchvision.models.resnet34(pretrained=False)
        # model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif modelname == "ResNet50":
        model = torchvision.models.resnet50(pretrained=False)
        # model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif modelname == "ResNet101":
        model = torchvision.models.resnet101(pretrained=False)
        # model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif modelname == "ResNet152":
        model = torchvision.models.resnet152(pretrained=False)
        # model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    else:
        raise Exception("{} is undefined".format(modelname))
    # log = logger if logger is not None else logging
    flag = False  # load successfully when only flag is true
    desc = None
    if pretrained_pth:
        if os.path.isfile(pretrained_pth):
            checkpoint = torch.load(pretrained_pth, weights_only=True)
            print("预训练路径和modelkey：", pretrained_pth, model_key)
            # load parameters
            ckpt_model_state_dict = checkpoint[model_key]
            if consistency:  # model and ckpt_model_state_dict is consistent.
                model.load_state_dict(ckpt_model_state_dict)
                # log.info("load all the parameters of pre-trianed model.")
            else:  # load parameter of layer-wise, resnet18 should load 120 layer at head.
                ckp_keys = list(ckpt_model_state_dict)
                cur_keys = list(model.state_dict())
                len_ckp_keys = len(ckp_keys)
                len_cur_keys = len(cur_keys)
                model_sd = model.state_dict()

                ckp_keys = ckp_keys[:120]
                cur_keys = cur_keys[:120]
                for ckp_key, cur_key in zip(ckp_keys, cur_keys):
                    model_sd[cur_key] = ckpt_model_state_dict[ckp_key]

                model.load_state_dict(model_sd)
        else:
            print("预训练不存在")
    else:
        print("没有加载预训练")
    return model

class img_2d_encoder(nn.Module):
    def __init__(self):
        super(img_2d_encoder, self).__init__()
        self.img_encoder = load_pretrained_component('ckpts/imagemol/CGIP.pth','model_state_dict1')
    #image encoder
    def _forward_impl(self, x:torch.Tensor):
        x = self.img_encoder.conv1(x)
        x = self.img_encoder.bn1(x)
        x = self.img_encoder.relu(x)
        x = self.img_encoder.maxpool(x)
        x = self.img_encoder.layer1(x)
        x = self.img_encoder.layer2(x)
        x = self.img_encoder.layer3(x)
        x = self.img_encoder.layer4(x)
        x = self.img_encoder.avgpool(x)
        x = torch.flatten(x, 1)
        # x = self.img_encoder.fc(x)
        return x

    def forward(self, x):
        out = self._forward_impl(x)  
        return out

imagemol = img_2d_encoder()

print(123)
img_transformer_test = transforms.Compose([transforms.CenterCrop(224), transforms.ToTensor()])
label_df = pd.read_csv(f'{data}/drugid_smiles.csv')
drug_id = label_df['id']
# 初始化一个字典来存储结果
results_dict = {}
print("start:")
for id in drug_id:
    img_path = f'{data}/2d_img/{id}.png'
    img = Image.open(img_path)
    img = img_transformer_test(img).unsqueeze(0)
    x = imagemol(img)
    x=x.squeeze(0)
    array = x.detach().numpy()
    # 打印结果
    print("Shape of concatenated array:", array.shape)
    # 将结果存储到字典中
    results_dict[id] = array

print(results_dict)
np.save(f'{data}/2dimagemolEmbed.npy',results_dict)
print("2d_feature done")


print("all done")