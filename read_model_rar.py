import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from generator import Generator
from classify import *
from utils import *
from SAC import Agent
from attack import inversion

if __name__ == "__main__":
    # model_name = "VGG16"
    model_name = "Face.evoLVe"
    max_episodes = 40000
    max_step = 1
    seed = 42
    alpha = 0
    n_classes = 1000
    z_dim = 100
    n_target = 1000

    print("Target Model : " + model_name)
    G = Generator(z_dim)
    G = nn.DataParallel(G).cuda()
    G = G.cuda()
    ckp_G = torch.load('weights/CelebA.tar')['state_dict']
    load_my_state_dict(G, ckp_G)
    G.eval()

    if model_name == "VGG16":
        # T = VGG16(n_classes)
        path_T = './weights/VGG16.tar'
    elif model_name == 'ResNet-152':
        # T = IR152(n_classes)
        path_T = './weights/ResNet-152.tar'
    elif model_name == "Face.evoLVe":
        # T = FaceNet64(n_classes)
        path_T = './weights/Face.evoLVe.tar'

    # T = torch.nn.DataParallel(T).cuda()
    ckp_T = torch.load(path_T)
    # T.load_state_dict(ckp_T['state_dict'], strict=False)
    # T.eval()

    print("\n\n\n")


    print(ckp_T.keys())
    # print(ckp_T['state_dict'].keys())
    # keys = ckp_T['state_dict'].keys()
    for k, v in ckp_T['state_dict'].items():
        print(k)



    # for k, v in ckp_G.items():
    #     print(k)
