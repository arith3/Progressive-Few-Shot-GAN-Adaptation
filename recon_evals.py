import argparse
from time import localtime, strftime
import time
import torch_fidelity
import random
import torch
import torch.nn as nn
from torch.nn import functional as F
from PIL import Image
import PIL

from tqdm import tqdm
import sys
import cv2 
import shutil
import numpy as np
import matplotlib.pyplot as plt
import easydict
import json
import os
from torchvision import utils
import torchvision.transforms as transforms
import torchvision.models as models

from model import Generator, Extra, Projection_module, Projection_module_church
from model import Patch_Discriminator as Discriminator  # , Projection_head
from dataset import MultiResolutionDataset

import lpips
import feat_cluster
from pytorch_fid import fid_score, inception
from torchvision.models.inception import inception_v3
from scipy.stats import entropy
import copy

import torch.multiprocessing as mp
from functools import partial

def requires_grad(model, flag=True):
    for name, p in model.named_parameters():
        p.requires_grad = flag

def get_noise(args):
    if args.noise:
        noise = torch.load(args.noise).to(args.device)
        print("noise loaded : " + args.noise, noise.shape)
    else:
        noise = torch.randn(args.n_sample, args.latent, device=args.device)
        print("noise generated : ", noise.shape)
    return noise

def load_model(ckpt, parallel, args):
    G = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(args.device)
    
    checkpoint = torch.load(ckpt)
    if parallel == True:
        G = nn.parallel.DataParallel(G)
        G.load_state_dict(checkpoint['g_ema'], strict=False)
    else:
        G.load_state_dict(checkpoint['g_ema'], strict=False)
        G = nn.parallel.DataParallel(G)
    G.eval()

    if 'd' in checkpoint:
        print("%s has discriminator." % ckpt)
        D = Discriminator(
            args.size, channel_multiplier=args.channel_multiplier
        ).to(args.device)
        if parallel == True:
            D = nn.parallel.DataParallel(D)
            D.load_state_dict(checkpoint['d'], strict=False)
        else:
            D.load_state_dict(checkpoint['d'], strict=False)
            D = nn.parallel.DataParallel(D)
        D.eval()
        del checkpoint
        return G, D
    print("%s has no discriminator." % ckpt)

    del checkpoint
    return G

def get_imgs_from_model(model, z, use_proj, args):
    if args.truncation < 1:
        with torch.no_grad():
            mean_latent = model.mean_latent(args.truncation_mean)
    else:
        mean_latent = None
    with torch.no_grad():
        model.eval()
        requires_grad(model, False)
        if use_proj:
            Proj_module = Projection_module(args)
            w = [model.module.style(z)]
            w = [Proj_module.modulate(item) for item in w]
            sample, _ = model(w, input_is_latent=True, randomize_noise=False)
        else:
            sample, _ = model([z])

        return sample

if __name__ == "__main__":
    start_time = time.time()
    mp.set_start_method('spawn')
    
    parser = argparse.ArgumentParser()
    
    # default
    parser.add_argument("--size", type=int, default=256) # image size
    parser.add_argument("--latent", type=int, default=512)
    parser.add_argument("--n_mlp", type=int, default=8)
    parser.add_argument("--channel_multiplier", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--noise", type=int, default=None)
    parser.add_argument("--truncation", type=int, default=1)
    parser.add_argument("--truncation_mean", type=int, default=4096)
    parser.add_argument("--n_sample", type=int, default=10000)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--n_process", type=int, default=10) # number of process
    
    # required
    #parser.add_argument("--target_name", type=str, required=True)
    
    parser.add_argument("--source_ckpt", type=str, required=True)
    parser.add_argument("--source_key", type=str, required=True)
    parser.add_argument("--model_ckpt", type=str, required=None)

    parser.add_argument("--sample_z", type=str, default=None)

    # for intra LPIPS
    parser.add_argument("--train_imgs", type=str)
    
    # Donot calc FID if it is None.
    parser.add_argument("--test_imgs", type=str, default=None)
    
    # w space for RSSA
    parser.add_argument("--latent_dir", type=str, default=None)
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--task", type=int, default=10)
    
    parser.add_argument("--skip_gen_src", type=bool, default=False)
    parser.add_argument("--no_parallel", dest='parallel', action='store_false')
    parser.add_argument("--parallel", dest='parallel', action='store_true')
    parser.add_argument("--use_identity", action='store_true')
    
    parser.add_argument("--target", type=str, default=None)

    cossim = nn.CosineSimilarity(dim=1)
    ceLoss = nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean')

    ## logging
    args = parser.parse_args()
    unique_id = int(time.time())
    
    metric_results = copy.deepcopy(args)

    # make out filename
    args.postfix = args.model_ckpt.split("/")[-1].split(".")[0]
    if args.postfix == 'final':
        args.postfix = args.model_ckpt.split("/")[-2]
    else:
        try:
            intValue = int(args.postfix)
            print(intValue)
            args.postfix = args.model_ckpt.split("/")[-2]
        except ValueError as verr:
            pass
        except Exception as ex:
            pass
    print("Eval : ", args.postfix)
    
    if args.target == None:
        args.target = args.postfix

    if args.seed != None:
        random_seed = args.seed
        torch.manual_seed(random_seed) 
        np.random.seed(random_seed) 
        random.seed(random_seed)
    
    # make out directory
    args.out_root = os.path.join(args.out, "_Reconstruction")
    if not os.path.exists(args.out_root):
        os.makedirs(args.out_root)

    ## initialization
    noise = get_noise(args)
    id_noise = torch.randn(1, args.latent, device=args.device)
    loss_fn_vgg = lpips.PerceptualLoss(
        model="net-lin", net="vgg", use_gpu=True
    )

    srcG, srcD = load_model(args.source_ckpt, parallel=False, args=args)
    recG = load_model(args.model_ckpt, parallel=args.parallel, args=args)
    if type(recG) == tuple:
        recG, recD = recG
    lowp, highp = 0, 1
    srcD.module.setConfig(extra=None, flag=0, p_ind=np.random.randint(lowp, highp))

    batch = args.batch
    print("Eval : Similarity")
    cossim_score = np.array([])
    LPIPS_score = np.array([])
    MSE_score = np.array([])
    with torch.no_grad():
        for start in tqdm(range(0,args.n_sample,args.batch)):
            end = start + args.batch
            samples_src = get_imgs_from_model(srcG, noise[start:end], False, args)
            if args.use_identity:
                samples_rec = get_imgs_from_model(srcG, id_noise.repeat(args.batch,1), False, args)
            else:
                samples_rec = get_imgs_from_model(recG, noise[start:end], args.latent_dir!=None, args)

            score = loss_fn_vgg(samples_src, samples_rec)
            LPIPS_score = np.append(LPIPS_score, score.cpu().numpy())
            score = nn.MSELoss()(samples_src, samples_rec)
            MSE_score = np.append(MSE_score, score.cpu().numpy())
            pred_src, feat_src = srcD(samples_src)
            pred_rec, feat_rec = srcD(samples_rec)
            score = cossim(feat_src[0].view(args.batch, -1), feat_rec[0].view(args.batch, -1))
            cossim_score = np.append(cossim_score, score.cpu().numpy())
        print("Average feature similarity score mean[%f], std[%f]" % (cossim_score.mean(), cossim_score.std()))
        print("Average feature LPIPS score mean[%f], std[%f]" % (LPIPS_score.mean(), LPIPS_score.std()))
        print("Average feature MSE score mean[%f], std[%f]" % (MSE_score.mean(), MSE_score.std()))
        metric_results.cossim_score = {"mean":cossim_score.mean().item(),"std":cossim_score.std().item()}
        metric_results.LPIPS_score = {"mean":LPIPS_score.mean().item(),"std":LPIPS_score.std().item()}
        metric_results.MSE_score = {"mean":MSE_score.mean().item(),"std":MSE_score.std().item()}

    batch = args.batch
    print("Eval : ic_loss")
    ic_loss = []
    with torch.no_grad():
        pbar = tqdm(range(0,args.n_sample))
        for i in pbar:
            if args.use_identity:
                samples_rec = get_imgs_from_model(srcG, id_noise, False, args)
            else:
                samples_rec = get_imgs_from_model(recG, noise[i].unsqueeze(dim=0), args.latent_dir!=None, args)

            pred_rec, feat_rec = srcD(samples_rec)
            feat_rec = feat_rec[0].repeat(args.n_sample,1,1,1).view(args.n_sample, -1)
            del pred_rec
            feat_src_list = []
            for start in range(0,args.n_sample,batch):
                end = start + batch
                samples_src = get_imgs_from_model(srcG, noise[start:end], False, args)
                pred_src, feat_src = srcD(samples_src)
                feat_src_list.append(feat_src[0].view(batch, -1))

            feat_src_list = torch.stack(feat_src_list, 0).view(args.n_sample, -1)
            score = cossim(feat_src_list, feat_rec).unsqueeze(0)
            score = ceLoss(score, torch.LongTensor([i]).to(args.device))
            ic_loss.append(score)
            pbar.set_description("CELoss %f" % score.item())

        ic_loss = torch.stack(ic_loss, 0)
        print(ic_loss.mean())
        metric_results.intra_cluster_CELoss = {"mean":ic_loss.mean().item(),"std":ic_loss.std().item()}

    ## Save results
    log_path = "%s_%d_%d.json" % (args.postfix, args.n_sample, unique_id)
    full_path = os.path.join(args.out_root, log_path)
    print(full_path)
    with open(full_path, 'w') as f:
        json.dump(metric_results.__dict__, f, indent=2)
            


