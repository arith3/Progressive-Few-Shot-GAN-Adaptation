import argparse
import easydict
import math
import random
import os
import sys
import numpy as np
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
import torch.nn.init as init
from torch.utils import data
import torch.distributed as dist
import torchvision
from torchvision import transforms, utils
from torchvision.ops import RoIAlign
from tqdm import tqdm
import viz
from copy import deepcopy
import numpy
import json
import cv2
from torchvision import models, transforms

import torchvision.transforms.functional as TF
import torchvision.transforms.functional_pil as F_pil
import torchvision.transforms.functional_tensor as F_t

import lpips

try:
    import wandb

except ImportError:
    wandb = None


from model import Generator, Extra, Projection_module, Sobel
from model import Patch_Discriminator as Discriminator  # , Projection_head
from dataset import MultiResolutionDataset
from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)
from non_leaking import augment

from DiffAugment_pytorch import DiffAugment

from da_metric import MetricProcess, make_metric_result

def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


def requires_grad(model, flag=True):
    for name, p in model.named_parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch

def d_logistic_loss(real_pred, fake_pred, src_pred=None):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    if src_pred == None:
        return real_loss.mean() + fake_loss.mean()
    else:
        return real_loss.mean() + fake_loss.mean() * 0.5 + F.softplus(src_pred).mean() * 0.5

def d_r1_loss(real_pred, real_img):
    grad_real, = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = grad_real.pow(2).reshape(
        grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty

def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred)
    return loss.mean()

def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )
    grad, = autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True
    )
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * \
        (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths


def make_noise(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)

    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)

    return noises


def mixing_noise(batch, latent_dim, prob, device):
    if prob > 0 and random.random() < prob:
        return make_noise(batch, latent_dim, 2, device)

    else:
        return [make_noise(batch, latent_dim, 1, device)]


def set_grad_none(model, targets):
    for n, p in model.named_parameters():
        if n in targets:
            p.grad = None


def get_subspace(args, init_z, vis_flag=False):
    std = args.subspace_std
    bs = args.batch if not vis_flag else args.n_sample
    init_z = init_z.clone()
    ind = np.random.randint(0, init_z.size(0), size=bs)
    z = init_z[ind]  # should give a tensor of size [batch_size, 512]
    z = normalize_noise(z, std)
    return z

def normalize_noise(z, std):
    for i in range(z.size(0)):
        for j in range(z.size(1)):
            z[i][j].data.normal_(z[i][j].item(), std)
    return z

def run_D(args, img, d, extra, flag, p_ind, real=False):
    if args.diff:
        img = DiffAugment(img, policy='color,translation,cutout')
    d.module.setConfig(extra=extra.to(device), flag=flag, p_ind=p_ind, real=real)
    d = d.to(device)
    logits = d(img.to(device))
    return logits

def get_images_from_dataset(dataset, indexes):
    images = None
    for idx in indexes:
        if images == None:
            images = dataset[idx].unsqueeze(dim=0)
        else:
            images = torch.cat([images, dataset[idx].unsqueeze(dim=0)], dim=0)
    return images

def noise_normalize_(noises):
    for noise in noises:
        mean = noise.mean()
        std = noise.std()

        noise.data.add_(-mean).div_(std)


def get_lr(t, initial_lr, rampup=0.05, rampdown=0.25):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    return initial_lr * lr_ramp


def latent_noise(latent, strength):
    noise = torch.randn_like(latent) * strength

    return latent + noise

def calcOpticalFlow(args, source, target):
    with torch.no_grad():
        result = None
        (b, c, w, h) = source.shape
        for i in range(b):
            _prev = source[i].cpu().detach().permute(1,2,0).numpy().copy() * 255.0
            _prev = cv2.normalize(_prev, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            if c >= 3:
                _prev = cv2.cvtColor(_prev, cv2.COLOR_BGR2GRAY)

            _next = target[i].cpu().detach().permute(1,2,0).numpy().copy() * 255.0
            _next = cv2.normalize(_next, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            if c >= 3:
                _next = cv2.cvtColor(_next, cv2.COLOR_BGR2GRAY)

            flow = cv2.calcOpticalFlowFarneback(_prev, _next, None,
                                            pyr_scale=0.5, levels=3,
                                            winsize=int(args.winsize),iterations=5,
                                            poly_n=5,poly_sigma=1.1,
                                            flags=0)
            if result == None:
                result = torch.tensor(flow).permute(2,0,1).unsqueeze(0)
            else:
                result = torch.cat([result,torch.tensor(flow).permute(2,0,1).unsqueeze(0)],dim=0)
            del _prev, _next
    return result

def calcAvgFlow(flow):
    blur = torchvision.transforms.GaussianBlur(kernel_size=5, sigma=1.0)
    blurFlow = blur(flow)
    center = flow.mean()
    flow = torch.where(flow < center, flow, blurFlow)
    return flow

def makeFlowSearchingMap(args, size):
    scale = 1.0
    center = [args.size * 0.5, args.size * 0.5]
    mat = []
    
    for y in range(size):
        for x in range(size):
            trans = [(x - size//2) * 0.5, (y - size//2) * 0.5]
            m = TF._get_inverse_affine_matrix(center, 0.0, trans, scale, [0.0,0.0])
            mat.append(m)
    return mat

def searchMatchedFeat(fs, ft, mat):
    criterion = nn.CosineSimilarity(dim=1)
    result = []
    for m in mat:
        fs_h = F_t.affine(fs, matrix=m)
        coff = criterion(fs_h, ft).float()
        result.append(coff)
        del fs_h, coff
        
    p = torch.stack(result, dim=1)
    del result
    return p

def calcFeatFlow(size, source, target, flowSearchingMap):
    result = None
    blur = torchvision.transforms.GaussianBlur(kernel_size=3, sigma=1.0)
    in2d = nn.InstanceNorm2d(128)
    with torch.no_grad():
        flow11 = searchMatchedFeat(in2d(source[11]), in2d(target[11]), flowSearchingMap)
        flow12 = searchMatchedFeat(in2d(source[12]), in2d(target[12]), flowSearchingMap)
        flow = (flow11 + flow12).argmax(dim=1)
        del flow11, flow12
    
    h = size // 2
    x = ((flow % size)-h).float() * 0.5
    y = ((flow // size)-h).float() * 0.5
    result = torch.stack([x, y], dim=1)
    #result = blur(result)
    
    del x, y, flow
    return result

def get_matched_values(indices, flow, fs, ft):
    ra1 = RoIAlign(1, spatial_scale=1, sampling_ratio=-1)
    (b,c,w,h) = ft.shape
    target_values = []
    xzip = indices %  w
    yzip = indices // w
    boxes = []
    for x,y in zip(xzip, yzip):
        xa = x + flow[0,y,x]
        ya = y + flow[1,y,x]
        boxes.append(torch.tensor([xa-.5,ya-.5,xa+0.5,ya+0.5]))
    boxes = torch.stack(boxes, dim=0).to(device)
    target_values = ra1(ft, [boxes]).view(indices.size(0), c)
    return target_values

def roundedM(x, _max, _scale):
    #x = np.min([_max, x])
    x = (x / _max)
    x = _max * (x - (x**2) / 2) * _scale
    x = np.max([0.0, x])
    return int(np.round(x))

def draw_flow(img, flow, step=16):
    h,w = img.shape[:2]
    y,x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx,fy = flow.permute(1,2,0).numpy()[y,x].T
    line = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1,2,2)

    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    cv2.polylines(vis, lines, 0, (0,255,255), lineType=cv2.LINE_AA)
    
    for (x1,y1), (_x2,y2) in lines:
        cv2.circle(vis, (x1,y1), 1, (0,128,255), -1, lineType=cv2.LINE_AA)

    return vis

def train(args, loader, generator, discriminator, extra, extra_ema, g_optim, d_optim, e_optim, g_ema, d_ema, device, g_source, d_source):
    
    loss_dict_extra = {}
    loss_dict_extra["gan"] = np.array([])
    loss_dict_extra["rel"] = np.array([])
    loss_dict_extra["lfc"] = np.array([])
    loss_dict_extra["cutoff"] = np.array([])
    
    loader = sample_data(loader)

    imsave_path = os.path.join('outputs/samples', args.exp)
    model_path = os.path.join('checkpoints', args.exp)

    if not os.path.exists(imsave_path):
        os.makedirs(imsave_path)
        
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    
    config_path = imsave_path + "/_args.json"
    with open(config_path, 'w') as f:
        json.dump(args.__dict__, f, indent=2)
        f.close()
        
    # this defines the anchor points, and when sampling noise close to these, we impose image-level adversarial loss (Eq. 4 in the paper)
    init_z = torch.randn(args.n_train, args.latent, device=device)
    
    sfm = nn.Softmax(dim=1)
    kl_loss = nn.KLDivLoss()
    l1_loss = nn.L1Loss()
    sim = nn.CosineSimilarity()
    avgpool = nn.AvgPool2d(kernel_size=2,stride=2)
    if args.kernal_size > 0:
        gaussianBlur = torchvision.transforms.GaussianBlur(kernel_size=args.kernal_size, sigma=1.0)
    gaussianBlurPred = torchvision.transforms.GaussianBlur(kernel_size=3, sigma=1.0)
    in2d= nn.InstanceNorm2d(512)

    mean_path_length = 0

    d_loss_val = 0
    r1_loss = torch.tensor(0.0, device=device)
    g_loss_val = 0
    path_loss = torch.tensor(0.0, device=device)
    path_lengths = torch.tensor(0.0, device=device)
    mean_path_length_avg = 0
    loss_dict = {}

    g_module = generator
    d_module = discriminator
    g_ema_module = g_ema.module

    accum = 0.5 ** (32 / (10 * 1000))
    accum_pa = args.k * (0.5 ** (32 / (10 * 1000)))
    ada_augment = torch.tensor([0.0, 0.0], device=device)
    ada_aug_p = args.augment_p if args.augment_p > 0 else 0.0
    ada_aug_step = args.ada_target / args.ada_length
    r_t_stat = 0

    # this defines which level feature of the discriminator is used to implement the patch-level adversarial loss: could be anything between [0, args.highp] 
    lowp, highp = 0, args.highp

    # the following defines the constant noise used for generating images at different stages of training
    sample_z = torch.randn(args.n_sample, args.latent, device=device)

    requires_grad(g_source, False)
    requires_grad(d_source, False)
    sub_region_z = get_subspace(args, init_z, vis_flag=True)
    if args.use_ema:
        if args.extra_ema and args.extra_key == "train":
            extra_optim = torch.optim.Adam(extra.parameters(),lr=1e-3,betas=(0.9, 0.999), eps=1e-08)
            scheduler = optim.lr_scheduler.StepLR(extra_optim, step_size=1, gamma=0.9995)
            extra_pbar = tqdm(range(args.extra_step), initial=0, dynamic_ncols=True, smoothing=0.01)
            for idx in extra_pbar:
                z = torch.randn(args.feat_const_batch, args.latent, device=device)
                real, _ = g_source([z])
                fake = next(loader).to(device)
                pred_real, feat_real = run_D(args, real, d_source, extra=extra, flag=1, p_ind=np.random.randint(lowp, highp))
                pred_fake, feat_fake = run_D(args, fake, d_source, extra=extra, flag=1, p_ind=np.random.randint(lowp, highp))
                loss = F.softplus(-pred_real).mean() + F.softplus(pred_fake).mean()
                extra_optim.zero_grad()
                loss.backward()
                extra_optim.step()
                scheduler.step()
                extra_pbar.set_description(
                    (
                        f"loss: {loss.item():.8f};"
                        f"lr: {extra_optim.param_groups[0]['lr']:.8f}"
                    )
                )
                del z, real, fake, pred_real, feat_real, pred_fake, feat_fake    
            del extra_optim, scheduler, extra_pbar
        accumulate(extra_ema, extra, 0)
    pbar = range(args.iter)
    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_iter,
                    dynamic_ncols=True, smoothing=0.01)

    if args.LFC:
        LFC_feat_range = args.LFC_layers.split(",")
        LFC_feat_range = list(map(int, LFC_feat_range))
        print("LFC Layers:", LFC_feat_range)
        
    if args.use_flow:
        flowSearchingMap = makeFlowSearchingMap(args, 9)
        
    if args.latent_dir != None:
        Proj_module = Projection_module(args)
        
    for idx in pbar:
        i = idx + args.start_iter

        which = i % args.subspace_freq # defines whether we sample from anchor region in this iteration or other

        if i > args.iter:
            print("Done!")
            break

        real_img = next(loader).to(device)

        requires_grad(generator, False)
        requires_grad(discriminator, True)
        requires_grad(extra, True)
        
        if which > 0:
            # sample normally, apply patch-level adversarial loss
            noise = mixing_noise(args.batch, args.latent, args.mixing, device)
        else:
            # sample from anchors, apply image-level adversarial loss
            noise = [get_subspace(args, init_z)]
            
        if args.latent_dir != None:
            
            w = [generator.style(item) for item in noise]
            w = [Proj_module.modulate(item) for item in w]
            fake_img, _ = generator(w, input_is_latent=True)
        else:
            fake_img, _ = generator(noise)
        
        if args.augment:
            real_img, _ = augment(real_img, ada_aug_p)
            fake_img, _ = augment(fake_img, ada_aug_p)
        
        real_pred, real_feat = run_D(args,
            real_img, discriminator, extra=extra, flag=which, p_ind=np.random.randint(lowp, highp), real=True)
        fake_pred, fake_feat = run_D(args,
            fake_img, discriminator, extra=extra, flag=which, p_ind=np.random.randint(lowp, highp))
        
        d_loss = d_logistic_loss(real_pred, fake_pred)
        
        del real_feat, fake_feat
        del fake_img
                    
        loss_dict["d"] = d_loss
        loss_dict["real_score"] = real_pred.mean()
        loss_dict["fake_score"] = fake_pred.mean()

        discriminator.zero_grad()
        extra.zero_grad()
        d_loss.backward()
        d_optim.step()
        e_optim.step()

        if args.augment and args.augment_p == 0:
            ada_augment += torch.tensor(
                (torch.sign(real_pred).sum().item(), real_pred.shape[0]), device=device
            )
            ada_augment = reduce_sum(ada_augment)

            if ada_augment[1] > 255:
                pred_signs, n_pred = ada_augment.tolist()

                r_t_stat = pred_signs / n_pred

                if r_t_stat > args.ada_target:
                    sign = 1

                else:
                    sign = -1

                ada_aug_p += sign * ada_aug_step * n_pred
                ada_aug_p = min(1, max(0, ada_aug_p))
                ada_augment.mul_(0)

        d_regularize = i % args.d_reg_every == 0

        if d_regularize:
            real_img.requires_grad = True

            real_pred, real_feat = run_D(args,
                real_img, discriminator, extra=extra, flag=which, p_ind=np.random.randint(lowp, highp))
            real_pred = real_pred.view(real_img.size(0), -1)
            real_pred = real_pred.mean(dim=1).unsqueeze(1)

            r1_loss = d_r1_loss(real_pred, real_img)

            discriminator.zero_grad()
            extra.zero_grad()
            (args.r1 / 2 * r1_loss * args.d_reg_every +
             0 * real_pred[0]).backward()

            d_optim.step()
            e_optim.step()
        loss_dict["r1"] = r1_loss

        requires_grad(generator, True)
        requires_grad(discriminator, False)
        requires_grad(extra, False)
        if which > 0:
            noise = mixing_noise(args.batch, args.latent, args.mixing, device)
        else:
            noise = [get_subspace(args, init_z)]

        if args.latent_dir != None:
            w = [generator.style(item) for item in noise]
            w = [Proj_module.modulate(item) for item in w]
            fake_img, _ = generator(w, input_is_latent=True)

        else:  fake_img, _ = generator(noise)

        if args.augment:
            fake_img, _ = augment(fake_img, ada_aug_p)

        fake_pred, fake_feat = run_D(args,
            fake_img, discriminator, extra=extra, flag=which, p_ind=np.random.randint(lowp, highp))
        g_loss = g_nonsaturating_loss(fake_pred)
        GAN_LOSS = g_loss.item()
        loss_dict_extra["gan"] = np.append(loss_dict_extra["gan"],[GAN_LOSS])

        if "real_feat" in locals():
            del real_feat
        del fake_img, fake_pred, fake_feat, real_img, real_pred
        
        if args.rel:
            # distance consistency loss
            with torch.set_grad_enabled(False):
                z = torch.randn(args.feat_const_batch, args.latent, device=device)
                feat_ind = numpy.random.randint(1, g_source.module.n_latent - 1, size=args.feat_const_batch)

                # computing source distances
                if args.use_rel_ema:
                    source_sample, feat_source = g_ema([z], return_feats=True)
                else:
                    source_sample, feat_source = g_source([z], return_feats=True)
                    
                del source_sample
                dist_source = torch.zeros(
                    [args.feat_const_batch, args.feat_const_batch - 1]).to(device)

                # iterating over different elements in the batch
                for pair1 in range(args.feat_const_batch):
                    tmpc = 0
                    # comparing the possible pairs
                    for pair2 in range(args.feat_const_batch):
                        if pair1 != pair2:
                            fa = feat_source[feat_ind[pair1]][pair1]
                            fc = feat_source[feat_ind[pair1]][pair2]
                            anchor_feat = torch.unsqueeze(fa.reshape(-1), 0)
                            compare_feat = torch.unsqueeze(fc.reshape(-1), 0)
                            dist_source[pair1, tmpc] = sim(
                                anchor_feat, compare_feat)
                            tmpc += 1
                            
                dist_source = sfm(dist_source)

            # computing distances among target generations
            if args.latent_dir != None:
                w = [generator.style(item) for item in [z]]
                w = [Proj_module.modulate(item) for item in w]
                target_sample, feat_target = generator(w, input_is_latent=True, return_feats=True)

            else:  target_sample, feat_target = generator([z], return_feats=True)
            
            del target_sample
            dist_target = torch.zeros(
                [args.feat_const_batch, args.feat_const_batch - 1]).to(device)

            # iterating over different elements in the batch
            for pair1 in range(args.feat_const_batch):
                tmpc = 0
                for pair2 in range(args.feat_const_batch):  # comparing the possible pairs
                    if pair1 != pair2:
                        fa = feat_target[feat_ind[pair1]][pair1]
                        fc = feat_target[feat_ind[pair1]][pair2]
                        anchor_feat = torch.unsqueeze(fa.reshape(-1), 0) # [1, x]
                        compare_feat = torch.unsqueeze(fc.reshape(-1), 0)
                        dist_target[pair1, tmpc] = sim(anchor_feat, compare_feat)
                        tmpc += 1
            
            dist_target = sfm(dist_target)
            rel_loss = args.kl_wt * \
                kl_loss(torch.log(dist_target), dist_source) # distance consistency loss 
            g_loss = g_loss + rel_loss
            REL_LOSS = rel_loss.item()
            loss_dict_extra["rel"] = np.append(loss_dict_extra["rel"],[REL_LOSS])

            del z

        
        if args.LFC:
            if args.winsize < 32:
                args.winsize *= args.winsize_inc
            z = torch.randn(args.feat_const_batch, args.latent, device=device)
            if args.latent_dir!=None:
                w = [generator.style(item) for item in [z]]
                w = [Proj_module.modulate(item) for item in w]
                fake_img, _ = generator(w, input_is_latent=True)
                sample_source, feat_source = g_ema(w, input_is_latent=True, return_feats=True) if args.use_ema else g_source(w, input_is_latent=True)
                sample_target, feat_target = generator(w, input_is_latent=True, return_feats=True)
            else:  
                sample_source, feat_source = g_ema([z], return_feats=True) if args.use_ema else g_source([z], return_feats=True)
                sample_target, feat_target = generator([z], return_feats=True)
            del z

            if args.use_flow:
                flow = calcOpticalFlow(args, sample_source, sample_target)
                
                flow_w = torch.absolute(flow).sum(dim=1).view(-1)
                cut_off = torch.quantile(flow_w, torch.tensor([args.cutoff])).to(device)
                loss_dict_extra["cutoff"] = np.append(loss_dict_extra["cutoff"],[cut_off.item()])
            
            inner_source_loss = []
            inner_target_loss = []
            
            ##Sobel
            sobel_filter = Sobel().to(device)
            sample_gray = torchvision.transforms.Grayscale(num_output_channels=1)(sample_target) #sample_source?
            sample_sobel = sobel_filter(sample_gray)
            sample_sobel = torchvision.transforms.GaussianBlur(kernel_size=(args.blur), sigma=(args.blur))(sample_sobel)

            for b in range(args.feat_const_batch):
                inner_source_loss.append(None)
                inner_target_loss.append(None)
            
            for feat_idx in range(len(LFC_feat_range)):
                layer = LFC_feat_range[feat_idx]
                fs_b = feat_source[layer]
                ft_b = feat_target[layer]
                    
                feat_size = fs_b.size(-1)
                if args.use_pred:
                    with torch.no_grad():
                        pred_map = sample_sobel
                        if pred_map.size(-1) > feat_size:
                            pred_map = nn.AdaptiveAvgPool2d((feat_size,feat_size))(pred_map)
                        elif pred_map.size(-1) < feat_size:
                            pred_map = nn.Upsample((feat_size,feat_size), mode='bilinear')(pred_map)

                        pred_map = nn.ReLU()(pred_map)
                        pred_map = pred_map.flatten(start_dim=1)
                        pred_map = torch.nan_to_num(pred_map, nan=0.0)
                
                if args.use_flow:
                    (fb, fc, fh, fw) = flow.shape
                    scale = feat_size / fw
                    curr_flow_b = nn.AdaptiveAvgPool2d((feat_size,feat_size))(flow) * scale
                    
                for sample_idx in range(args.feat_const_batch):
                    fs = fs_b[sample_idx]
                    ft = ft_b[sample_idx]
                    
                    (c, h, w) = fs.shape
                    
                    if args.kernal_size > 0:
                        blur_sigma = 2.0 - w / 256.0
                        fs = torchvision.transforms.GaussianBlur(kernel_size=args.kernal_size, sigma=blur_sigma)(fs)
                        ft = torchvision.transforms.GaussianBlur(kernel_size=args.kernal_size, sigma=blur_sigma)(ft)
                    
                    m = 2 * feat_size
                    if args.use_pred:
                        pred = pred_map[sample_idx]
                        if args.use_flow and args.use_flow_pred:
                            flow_w = torch.absolute(curr_flow_b[sample_idx]).sum(dim=0).view(-1)
                            pred[flow_w > cut_off.item() * scale] = 0.0
                            pred = (pred - torch.min(pred)) / (torch.max(pred) - torch.min(pred))
                        indices_l = torch.multinomial(pred, m, replacement=True)
                        indices_r = torch.randint(low=0, high=w*h, size=(m,), device=device)
                    else:
                        indices_l = torch.randint(low=0, high=w*h, size=(m,), device=device)
                        indices_r = torch.randint(low=0, high=w*h, size=(m,), device=device)
                       
                    if args.use_flow:
                        ft_l = get_matched_values(indices_l, curr_flow_b[sample_idx], fs.unsqueeze(0), ft.unsqueeze(0))
                        ft_r = get_matched_values(indices_r, curr_flow_b[sample_idx], fs.unsqueeze(0), ft.unsqueeze(0))
                    else:
                        ft = ft.reshape(ft.size(0), -1)
                        ft_l = torch.index_select(ft, dim=1, index=indices_l).permute(1,0)
                        ft_r = torch.index_select(ft, dim=1, index=indices_r).permute(1,0)
                        
                    fs = fs.reshape(fs.size(0), -1) # [channel, w * h]
                    fs_l = torch.index_select(fs, dim=1, index=indices_l).permute(1,0)
                    fs_r = torch.index_select(fs, dim=1, index=indices_r).permute(1,0)

                    if inner_source_loss[sample_idx] == None:
                        inner_source_loss[sample_idx] = sim(fs_l, fs_r)
                        inner_target_loss[sample_idx] = sim(ft_l, ft_r)
                    else:
                        inner_source_loss[sample_idx] = torch.cat([inner_source_loss[sample_idx], sim(fs_l, fs_r)], dim=0)
                        inner_target_loss[sample_idx] = torch.cat([inner_target_loss[sample_idx], sim(ft_l, ft_r)], dim=0)

                    del indices_l, indices_r
                    del fs_l, ft_l, fs_r, ft_r
                    
                    if "flow_w" in locals():
                        del flow_w

                if "curr_flow_b" in locals():
                        del curr_flow_b
                if "pred_map" in locals():
                    del pred_map
            if "pred_source" in locals():
                del pred_source
            if "pred_target" in locals():
                del pred_target
                
            inner_source_loss = torch.stack(inner_source_loss)
            inner_target_loss = torch.stack(inner_target_loss)
            
            
            if args.loss == "L1":
                inner_loss = l1_loss(sfm(inner_target_loss), sfm(inner_source_loss)) * 10.0 * args.kl_wt * args.LFCw
            else:
                inner_loss = (kl_loss(torch.log(sfm(inner_target_loss)), sfm(inner_source_loss))) * 400.0 * args.kl_wt * args.LFCw
            
            LFC_LOSS = inner_loss.item()
            loss_dict_extra["lfc"] = np.append(loss_dict_extra["lfc"],[LFC_LOSS])
            g_loss += inner_loss

            if "flow" in locals():
                del flow
            
        loss_dict["g"] = g_loss
        generator.zero_grad()
        g_loss.backward()
        g_optim.step()
        
        g_regularize = i % args.g_reg_every == 0


        if g_loss.item() > 20.0:
            log = "\r\n %i: g_loss(%f), GAN_LOSS(%f)" % (i, g_loss, GAN_LOSS)
            if "REL_LOSS" in locals():
                log += ", REL_LOSS(%f)" % REL_LOSS
            if "LFC_LOSS" in locals():
                log += ", LFC_LOSS(%f)" % LFC_LOSS
            print(log)

        # to save up space
        del g_loss, d_loss

        
        if 'inner_dist_loss' in locals(): 
            del inner_dist_loss 
        if 'inner_source_loss' in locals(): 
            del inner_source_loss 
        if 'inner_target_loss' in locals(): 
            del inner_target_loss 
            
        if 'anchor_feat' in locals(): 
            del anchor_feat
        if 'compare_feat' in locals(): 
            del compare_feat
        if 'rel_loss' in locals(): 
            del rel_loss
        if 'dist_source' in locals(): 
            del dist_source
        if 'dist_target' in locals(): 
            del dist_target
        if 'feat_source' in locals(): 
            del feat_source 
        if 'feat_target' in locals(): 
            del feat_target    
            
        if g_regularize:
            path_batch_size = max(1, args.batch // args.path_batch_shrink)
            noise = mixing_noise(
                path_batch_size, args.latent, args.mixing, device)
            fake_img, latents = generator(noise, return_latents=True)

            path_loss, mean_path_length, path_lengths = g_path_regularize(
                fake_img, latents, mean_path_length
            )

            generator.zero_grad()
            weighted_path_loss = args.path_regularize * args.g_reg_every * path_loss

            if args.path_batch_shrink:
                weighted_path_loss += 0 * fake_img[0, 0, 0, 0]

            weighted_path_loss.backward()

            g_optim.step()

            mean_path_length_avg = (
                reduce_sum(mean_path_length).item() / get_world_size()
            )

        
        loss_dict["path"] = path_loss
        loss_dict["path_length"] = path_lengths.mean()

        accumulate(g_ema_module, g_module, accum)
        
        if args.use_ema:
            accumulate(d_ema.module, d_module.module, accum_pa)
            accumulate(extra_ema.module, extra.module, accum_pa)
            
        loss_reduced = reduce_loss_dict(loss_dict)

        d_loss_val = loss_reduced["d"].mean().item()
        g_loss_val = loss_reduced["g"].mean().item()
        r1_val = loss_reduced["r1"].mean().item()
        path_loss_val = loss_reduced["path"].mean().item()
        real_score_val = loss_reduced["real_score"].mean().item()
        fake_score_val = loss_reduced["fake_score"].mean().item()
        path_length_val = loss_reduced["path_length"].mean().item()

        if 'z' in locals(): 
            del z
            
        if get_rank() == 0:
            if args.LFC == False:
                LFC_LOSS = 0
            pbar.set_description(
                (
                    f"i: {i:d}; l: {LFC_LOSS:.2f};"
                    f"d: {d_loss_val:.4f}; g: {g_loss_val:.4f}; r1: {r1_val:.4f}; "
                    #f"path: {path_loss_val:.4f}; mean path: {mean_path_length_avg:.4f}; "
                    #f"augment: {ada_aug_p:.4f}"
                )
            )
            

            if wandb and args.wandb:
                wandb.log(
                    {
                        "Generator": g_loss_val,
                        "Discriminator": d_loss_val,
                        "Augment": ada_aug_p,
                        "Rt": r_t_stat,
                        "R1": r1_val,
                        "Path Length Regularization": path_loss_val,
                        "Mean Path Length": mean_path_length,
                        "Real Score": real_score_val,
                        "Fake Score": fake_score_val,
                        "Path Length": path_length_val,
                    }
                )

            if i % args.img_freq == 0:
                with torch.no_grad():
                    with torch.set_grad_enabled(False):
                        g_ema.eval()
                        if args.latent_dir != None:
                            w = [g_ema.module.style(sample_z.data)]
                            w = [Proj_module.modulate(item) for item in w]
                            sample, _ = g_ema(w, input_is_latent=True)
                        else:
                            sample, _ = g_ema([sample_z.data])

                        utils.save_image(
                            sample,
                            f"%s/{str(i).zfill(6)}.png" % (imsave_path),
                            nrow=int(args.n_sample ** 0.5),
                            normalize=True,
                            range=(-1, 1),
                        )
                        del sample
                    np.save(f"%s/_loss_{str(i).zfill(6)}" % (imsave_path), loss_dict_extra)
                
            if (i % args.save_freq == 0) and (i > 0):
                if args.save_full_model:
                    save_models = {
                        "g_ema": g_ema.state_dict(),
                        # uncomment the following lines only if you wish to resume training after saving. 
                        #Otherwise, saving just the generator is sufficient for evaluations
                        "g": g_module.state_dict(),
                        "g_s": g_source.state_dict(),
                        "d": d_module.state_dict(),
                        "g_optim": g_optim.state_dict(),
                        "d_optim": d_optim.state_dict(),
                    }
                else:
                    save_models = {
                        "g_ema": g_ema.state_dict(),
                        #"latent": init_z,
                    }
                
                torch.save(
                    save_models,
                    f"%s/{str(i).zfill(6)}.pt" % (model_path),
                )


if __name__ == "__main__":
    device = "cuda"
    parser = argparse.ArgumentParser()

    torch.cuda.empty_cache()
    
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--iter", type=int, default=5002)
    parser.add_argument("--save_freq", type=int, default=1000)
    parser.add_argument("--img_freq", type=int, default=1000)
    parser.add_argument("--kl_wt", type=int, default=1000)
    parser.add_argument("--highp", type=int, default=1)
    parser.add_argument("--subspace_freq", type=int, default=4)
    parser.add_argument("--feat_ind", type=int, default=3)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--feat_const_batch", type=int, default=4)
    parser.add_argument("--n_sample", type=int, default=25)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--patch_size", type=int, default=4)
    parser.add_argument("--feat_res", type=int, default=128)
    parser.add_argument("--r1", type=float, default=10)
    parser.add_argument("--path_regularize", type=float, default=2)
    parser.add_argument("--path_batch_shrink", type=int, default=2)
    parser.add_argument("--d_reg_every", type=int, default=16)
    parser.add_argument("--g_reg_every", type=int, default=4)
    parser.add_argument("--mixing", type=float, default=0.9)
    parser.add_argument("--subspace_std", type=float, default=0.1)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--source_key", type=str, default='ffhq')
    parser.add_argument("--exp", type=str, default=None, required=True)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--channel_multiplier", type=int, default=2)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--augment", dest='augment', action='store_true')
    parser.add_argument("--no-augment", dest='augment', action='store_false')
    parser.add_argument("--augment_p", type=float, default=0.0)
    parser.add_argument("--ada_target", type=float, default=0.6)
    parser.add_argument("--ada_length", type=int, default=500 * 1000)
    parser.add_argument("--n_train", type=int, default=10)
    
    parser.add_argument("--no_rel", dest='rel', action="store_false")
    parser.add_argument("--diff", action="store_true")
    
    parser.add_argument("--LFCw", type=float, default=0.4)
    parser.add_argument("--LFC_layers", type=str, default="1,2,3,4,5,6,7,8,9,10")

    parser.add_argument("--in2d", action="store_true")
    parser.add_argument("--pred_noiseW", type=float, default=1.0)
    parser.add_argument("--m", type=float, default=2.0)
    parser.add_argument("--no_use_pixel", dest='use_pixel', action="store_false")
    parser.add_argument("--kernal_size", type=int, default=0)
    parser.add_argument("--blur_pred", action="store_true")
    parser.add_argument("--extra_ema", action="store_true")
    parser.add_argument("--extra_step", type=int, default=2000)
    parser.add_argument("--winsize", type=float, default=16)
    parser.add_argument("--winsize_inc", type=float, default=1.0002)
    
    parser.add_argument("--loss", type=str, default="kl")
    parser.add_argument('--task', type=int, default=10)
    
    # ablation study
    parser.add_argument("--extra_key", type=str, default='train')
    parser.add_argument("--LFC"      , action="store_true")
    parser.add_argument("--use_pred" , action="store_true")
    parser.add_argument("--use_ema"  , action="store_true")
    parser.add_argument("--k", type=float, default=1.0)
    
    parser.add_argument("--use_flow", action="store_true")
    
    parser.add_argument("--metric", dest='metric', action='store_true')
    parser.add_argument("--no_metric", dest='metric', action='store_false')

    parser.add_argument("--metric_source_path", type=str, default=None)
    parser.add_argument('--latent_dir', type=str, default=None)
    parser.add_argument('--exp_name', type=str, default=None)
   
    parser.add_argument("--use_rel_ema", action="store_true")

    parser.add_argument("--use_flow_pred", action="store_true")
    parser.add_argument("--use_cut_off", action="store_true")
    parser.add_argument("--save_full_model", action="store_true")
    parser.add_argument("--src_parallel", action="store_true")
        
    parser.add_argument("--blur", type=int, default=5)
    parser.add_argument("--cutoff", type=float, default=0.8)
    

    args = parser.parse_args()

    random_seed = 1 
    torch.manual_seed(random_seed) 
    random.seed(random_seed)
    
    n_gpu = 4
    args.distributed = n_gpu > 1
    
    args.latent = 512
    args.n_mlp = 8

    args.start_iter = 0

    generator = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    g_source = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    discriminator = Discriminator(
        args.size, channel_multiplier=args.channel_multiplier
    ).to(device)
    d_source = Discriminator(
        args.size, channel_multiplier=args.channel_multiplier
    ).to(device)
    
    d_ema = None
    extra_ema = None
    if args.use_ema and args.LFC:
        d_ema = Discriminator(
            args.size, channel_multiplier=args.channel_multiplier
        ).to(device)
        extra_ema = Extra().to(device)
        
    g_ema = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)

    extra = Extra().to(device)
    

    g_ema.eval()
    accumulate(g_ema, generator, 0)

    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

    g_optim = optim.Adam(
        generator.parameters(),
        lr=args.lr * g_reg_ratio,
        betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
    )
    d_optim = optim.Adam(
        discriminator.parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )

    e_optim = optim.Adam(
        extra.parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )
    
    module_source = ['landscapes', 'red_noise',
                     'white_noise', 'hands', 'mountains', 'handsv2']
    if args.distributed and args.src_parallel:
        geneator = nn.parallel.DataParallel(generator)
        g_source = nn.parallel.DataParallel(g_source)
        g_ema = nn.parallel.DataParallel(g_ema)
        
        if args.use_ema and args.LFC:
            d_ema = nn.parallel.DataParallel(d_ema)
            extra_ema = nn.parallel.DataParallel(extra_ema)
            
        discriminator = nn.parallel.DataParallel(discriminator)
        d_source = nn.parallel.DataParallel(d_source)

    if args.ckpt is not None:
        print("load model:", args.ckpt)
        assert args.source_key in args.ckpt
        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)
        try:
            ckpt_name = os.path.basename(args.ckpt)
            args.start_iter = int(os.path.splitext(ckpt_name)[0])

        except ValueError:
            pass

        print(ckpt.keys())

        generator.load_state_dict(ckpt["g"], strict=False)
        discriminator.load_state_dict(ckpt["d"], strict=False)
        g_ema.load_state_dict(ckpt["g_ema"], strict=False)
        if args.use_ema and args.LFC:
            d_ema.load_state_dict(ckpt["d"], strict=False)
        
        g_source.load_state_dict(ckpt["g"], strict=False)
        d_source.load_state_dict(ckpt["d"], strict=False)

        if 'g_optim' in ckpt.keys():
            g_optim.load_state_dict(ckpt["g_optim"])
        if 'd_optim' in ckpt.keys():
            d_optim.load_state_dict(ckpt["d_optim"])
            
        del ckpt
        
    if args.extra_key != "train":
        ckpt = torch.load(args.extra_key, map_location=lambda storage, loc: storage)
        extra.load_state_dict(ckpt, strict=False)
        
    if args.distributed:
        extra = nn.parallel.DataParallel(extra)
        if not args.src_parallel:
            geneator = nn.parallel.DataParallel(generator)
            g_source = nn.parallel.DataParallel(g_source)
            g_ema = nn.parallel.DataParallel(g_ema)
            
            if args.use_ema and args.LFC:
                d_ema = nn.parallel.DataParallel(d_ema)
                extra_ema = nn.parallel.DataParallel(extra_ema)
                
            discriminator = nn.parallel.DataParallel(discriminator)
            d_source = nn.parallel.DataParallel(d_source)
        
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )

    dataset = MultiResolutionDataset(args.data_path, transform, args.size)
    args.n_train = len(dataset) if len(dataset) > 4 else 1
    sampler=data_sampler(dataset, shuffle=True, distributed=False)
        
    loader = data.DataLoader(
        dataset,
        batch_size=args.batch,
        sampler=sampler,
        drop_last=True,
    )
    
    if get_rank() == 0 and wandb is not None and args.wandb:
        wandb.init(project="stylegan 2")

    del transform
    
    train(args, loader, generator, discriminator, extra, extra_ema, g_optim,
          d_optim, e_optim, g_ema, d_ema, device, g_source, d_source)

    del generator, g_source, discriminator, d_source, g_ema, extra, d_ema, extra_ema
    del g_optim, d_optim, e_optim
    
    if args.metric:
        for it in range(args.save_freq, args.iter, args.save_freq):
            make_metric_result(args, '{0:06d}'.format(it), device)
