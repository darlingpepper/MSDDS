import torch
import torch.utils.data as data
import json
import numpy as np
import pandas as pd

from PIL import Image
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

class SynergyBothEncoderDataset(data.Dataset):
    def __init__(self, dataset_name, i, type, split):
        csv = f'{dataset_name}/{split}/{i}/{type}.csv'
        self.dataset_name = dataset_name
        label_df = pd.read_csv(csv)
        self.drug_1_smiles = label_df['drug_1']
        self.drug_2_smiles = label_df['drug_2']
        self.context = label_df['cell']
        self.Y = label_df['label']
        self.len = len(self.Y)#具体有多少数据

        self.img_2d_embed = np.load(f"{dataset_name}/2dimagemolEmbed.npy", allow_pickle=True).item()
        self.img_3d_embed = np.load(f"{dataset_name}/3dvideomolEmbed.npy", allow_pickle=True).item()
        self.cell_features = np.load(f"{dataset_name}/cell.npy", allow_pickle=True).item()
    def __len__( self ):
        return self.len
    def __getitem__(self, index):
        synergyScore = self.Y[index]
        drug1 = self.drug_1_smiles[index]
        drug2 = self.drug_2_smiles[index]
        drug1_2d_features = self.img_2d_embed[drug1]
        drug2_2d_features = self.img_2d_embed[drug2]
        drug1_3d_features = self.img_3d_embed[drug1]
        drug2_3d_features = self.img_3d_embed[drug2]
        context_features=self.cell_features[self.context[index]]#从context中获取特征
        context_features = np.array(context_features)
        return [
                torch.LongTensor([synergyScore]),
                torch.FloatTensor([context_features]),
                torch.FloatTensor(drug1_2d_features),
                torch.FloatTensor(drug1_3d_features),
                torch.FloatTensor(drug2_2d_features),
                torch.FloatTensor(drug2_3d_features),
                ]
