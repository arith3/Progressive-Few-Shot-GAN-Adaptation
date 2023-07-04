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

def search(dirname):
    result = []
    filenames = list(filter(lambda x: not x.startswith('.'), os.listdir(dirname)))
    for filename in filenames:
        full_filename = os.path.join(dirname, filename)
        result.append(full_filename)
    return result

def requires_grad(model, flag=True):
    for name, p in model.named_parameters():
        p.requires_grad = flag

def print_duration(start):
    duration_time = (time.time() - start)
    total_minutes = duration_time // 60
    hours = total_minutes // 60
    minuts = total_minutes % 60
    seconds = duration_time % 60
    print("duration [ %i : %i : %i ]" % (hours, minuts, seconds))
    print("millisecond", (duration_time*1000) % 1000)
    return total_minutes

def reset_foler(path):
    if os.path.exists(path):
        try:
            shutil.rmtree(path)
        except OSError as e:
            print ("Error: %s - %s." % (e.filename, e.strerror))
    os.makedirs(path)
    
def get_noise(args):
    if args.noise:
        noise = torch.load(args.noise).to(args.device)
        print("noise loaded : " + args.noise, noise.shape)
    else:
        noise = torch.randn(args.n_sample, args.latent, device=args.device)
        print("noise generated : ", noise.shape)
    return noise

def get_sample_noise(args):
    if args.sample_z != None:
        noise = torch.load(args.sample_z).to(args.device)
        print("noise loaded : " + args.sample_z, noise.shape)
        return noise
    return None

def load_model(ckpt, parallel, args):
    model = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(args.device)
    checkpoint = torch.load(ckpt)
    if parallel == True:
        model = nn.parallel.DataParallel(model)
        model.load_state_dict(checkpoint['g_ema'], strict=False)
    else:
        model.load_state_dict(checkpoint['g_ema'], strict=False)
        model = nn.parallel.DataParallel(model)
    model.eval()

    del checkpoint
    return model


def generate_imgs_from_pt(ckpt, out, noise, parallel, args):
    batch = args.batch
    # ckpt
    model = load_model(ckpt, parallel, args)
    reset_foler(out)

    if args.latent_dir != None:
        Proj_module = Projection_module(args)
        print("!!!generation on w space!!!")

    if args.truncation < 1:
        with torch.no_grad():
            mean_latent = model.mean_latent(args.truncation_mean)
    else:
        mean_latent = None

    with torch.no_grad():
        model.eval()
        requires_grad(model, False)

        n_sample = noise.size(0)
        for start in tqdm(range(0,n_sample, batch), leave=False):
            if start + batch <= n_sample:
                end = start + batch
                if args.latent_dir != None:
                    w = [model.module.style(noise[start:end])]
                    w = [Proj_module.modulate(item) for item in w]
                    sample, _ = model(w, input_is_latent=True, randomize_noise=False)
                else:
                    sample, _ = model([noise[start:end]])
                for index in range(sample.shape[0]):
                    utils.save_image(
                        sample[index],
                        out + (f'/sample%03d.png' % (start+index)),
                        nrow=5,
                        normalize=True,
                        range=(-1, 1),
                        )
            else:
                for index in range(start, n_sample):
                    z = noise[index,:].unsqueeze(0)

                    if args.latent_dir != None:
                        w = [model.module.style(z)]
                        w = [Proj_module.modulate(item) for item in w]
                        sample, _ = model(w, input_is_latent=True, randomize_noise=False)
                    else:
                        sample, _ = model([z])
                        
                    utils.save_image(
                        sample,
                        out + (f'/sample%03d.png' % index),
                        nrow=5,
                        normalize=True,
                        range=(-1, 1),
                        )
                    
    del model

def generate_imgs_from_ds(ckpt, out, args):
    reset_foler(out)
    transform = transforms.Compose([])
    dataset = MultiResolutionDataset(ckpt, transform, args.size)
    _size = len(dataset) if len(dataset) >= 5 else 1
    for index in range(_size):
        name = out + (f'/sample%03d.png' % (index))
        dataset[index].save(name, "png")
    del dataset
        
def clustering_worker(i, model_imgs, fakes, centers, out, transform, loss_fn_vgg, device):
    with torch.no_grad():
        _from = os.path.join(model_imgs, fakes[i])
        fake = transform(Image.open(_from)).to(device).repeat(centers.size(0),1,1,1)
        scores = loss_fn_vgg(fake, centers)#.reshape(centers.size(0))
        _to = os.path.join(out, "c%d"%(torch.argmin(scores).item()))
        os.system("cp %s %s" %(_from, _to))
        del fake, scores
    return i

def make_cluster(train_imgs, model_imgs, out, args):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )

    loss_fn_vgg = lpips.PerceptualLoss(
        model="net-lin", net="vgg", use_gpu=True
    )
    
    n_sample = args.n_sample
    
    reset_foler(out)

    reals = list(filter(lambda x: not x.startswith('.'), os.listdir(train_imgs)))
    fakes = list(filter(lambda x: not x.startswith('.'), os.listdir(model_imgs)))

    centers=[]
    for i in range(len(reals)):
        current_path = os.path.join(out, "c%d"%(i))
        os.makedirs(current_path)
        real= os.path.join(train_imgs, reals[i])
        imgC= os.path.join(current_path, "center.png")
        os.system("cp %s %s" %(real, imgC))
        c = transform(Image.open(imgC)).to(args.device)
        centers.append(c)
    
    centers = torch.stack(centers)

    worker_fn = partial(clustering_worker, 
        model_imgs=model_imgs, fakes=fakes, centers=centers, out=out, 
        transform=transform, loss_fn_vgg=loss_fn_vgg, device=args.device)

    start_time = time.time()

    with mp.Pool(processes=args.n_process) as pool:
        #for i in tqdm(pool.map(worker_fn, range(n_sample))):
        #    pass
        p = pool.map(worker_fn, range(n_sample))

    # for i in tqdm(range(n_sample)):
    #     _from = os.path.join(model_imgs, fakes[i])
    #     fake = transform(Image.open(_from)).to(args.device).repeat(centers.size(0),1,1,1)
    #     scores = loss_fn_vgg(fake, centers)#.reshape(centers.size(0))
    #     _to = os.path.join(out, "c%d"%(torch.argmin(scores).item()))
    #     os.system("cp %s %s" %(_from, _to))
    #     del fake, scores

    print_duration(start_time)

    cluster_args = easydict.EasyDict({
        "baseline":"",
        "dataset": "",
        "root": out,
        "n_train": len(reals),
    })
    
    mean, std = feat_cluster.intra_cluster_dist(cluster_args)
    #feat_cluster.get_close_far_members(cluster_args)
    
    del centers
    del loss_fn_vgg

    torch.cuda.empty_cache()
    return mean.item(), std.item()

def inception_score(img_dir, batch_size=100, resize=False, splits=1):
    """Computes the inception score of the generated images imgs
    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    dat = []
    imgs = list(filter(lambda x: not x.startswith('.'), os.listdir(img_dir)))
    N = len(imgs)
    print('%d images to be evaluated' % N)
    for i in tqdm(range(N)):
        img = np.array(Image.open(os.path.join(img_dir,imgs[i])))
        img = torch.from_numpy(img)
        img = img.permute(2, 0, 1).unsqueeze(0)
        img = 2 * img / 255.0 - 1
        dat.append(img)
    
    dat = torch.cat(dat, dim=0)

    assert batch_size > 0
    assert N >= batch_size
    
    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).cuda()
    inception_model.eval()
    up = nn.Upsample(size=(299, 299), mode='bilinear').cuda()
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i in tqdm(range(int(N / batch_size))):
        batch = dat[i* batch_size: (i+1) * batch_size, :, :, :].cuda()
        
        #batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batch)

    # Now compute the mean kl-div
    split_scores = []

    for k in tqdm(range(splits)):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in tqdm(range(part.shape[0]), leave=False):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))
    
    IS_mean, IS_std = np.mean(split_scores), np.std(split_scores)
    print(IS_mean, IS_std)
    return IS_mean, IS_std

feature_layers = {'0': 'conv1_1',
                  '5': 'conv2_1',
                  '10': 'conv3_1',
                  '19': 'conv4_1',
                  '21': 'conv4_2',
                  '28': 'conv5_1'}
style_layers_dict = {'conv1_1':0.75,
                     'conv2_1':0.5,
                     'conv3_1':0.25,
                     'conv4_1':0.25,
                     'conv5_1':0.25}
content_layer = 'conv5_1'

#def gram_matrix(x, should_normalize=True):
#    (b, ch, h, w) = x.size()
#    features = x.view(b, ch, w * h)
#    features_t = features.transpose(1, 2)
#    gram = features.bmm(features_t)
#    if should_normalize:
#        gram /= ch * h * w
#    return gram

def gram_matrix(x):
    n, c, h, w = x.size()
    x = x.view(n*c, h*w)
    gram = torch.mm(x,x.t()) # 행렬간 곱셈 수행
    return gram

def get_features(x, model, layers):
    features = {}
    for name, layer in enumerate(model.children()): # 0, conv
        x = layer(x)
        if str(name) in layers:
            features[layers[str(name)]] = x
    return features

def get_content_loss(pred_features, target_features, layer):
    target = target_features[layer]
    pred = pred_features[layer]
    loss = F.mse_loss(pred, target)
    return loss

def get_style_loss(pred_features, target_features, style_layers_dict):
    loss = 0
    for layer in style_layers_dict:
        pred_fea = pred_features[layer]
        pred_gram = gram_matrix(pred_fea)
        n, c, h, w = pred_fea.shape
        target_gram = gram_matrix(target_features[layer])
        layer_loss = style_layers_dict[layer] * F.mse_loss(pred_gram, target_gram)
        loss += layer_loss / (n*c*h*w)
    return loss

def distance_from_source(source, target, args):
    transform = transforms.Compose(
        [transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),]
    )

    model_vgg = model_vgg = models.vgg19(pretrained=True).features.to("cuda" if args.device else "cpu").eval()
    
    style_result = []
    content_result = []

    source_path = search(source)
    target_path = search(target)

    size = np.min([len(source_path), len(target_path)])

    for i in tqdm(range(size), leave=False):
        source_img = torch.unsqueeze(transform(Image.open(source_path[i])),0).to(args.device)
        target_img = torch.unsqueeze(transform(Image.open(target_path[i])),0).to(args.device)
        source_feat = get_features(source_img, model_vgg, feature_layers)
        target_feat = get_features(target_img, model_vgg, feature_layers)

        style_loss = get_style_loss(source_feat, target_feat, style_layers_dict)
        content_loss = get_content_loss(source_feat, target_feat, content_layer)

        style_result.append(style_loss.detach().cpu().item())
        content_result.append(content_loss.detach().cpu().item())

        del source_feat, target_feat, style_loss, content_loss
    return np.array(style_result).mean(), np.array(content_result).mean()

def kid_score(test_imgs, model_imgs, args):
    imgs_test = list(filter(lambda x: not x.startswith('.'), os.listdir(test_imgs)))
    imgs_model = list(filter(lambda x: not x.startswith('.'), os.listdir(model_imgs)))

    print(len(imgs_test), len(imgs_model))
    metrics = torch_fidelity.calculate_metrics(
            cuda=args.device,
            input1=model_imgs,
            input2=test_imgs,
            isc=True,
            fid=True,
            kid=True,
            kid_subset_size=np.min([len(imgs_test), len(imgs_model)]),
            ppl=False,
        )
    print(metrics)
    return metrics
        
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
    parser.add_argument("--train_imgs", type=str, required=True)
    
    # Donot calc FID if it is None.
    parser.add_argument("--test_imgs", type=str, default=None)
    
    # w space for RSSA
    parser.add_argument("--latent_dir", type=str, default=None)
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--task", type=int, default=10)
    
    parser.add_argument("--skip_gen_src", type=bool, default=False)
    parser.add_argument("--no_parallel", dest='parallel', action='store_false')
    parser.add_argument("--parallel", dest='parallel', action='store_true')
    
    parser.add_argument("--target", type=str, default=None)

    ## logging
    args = parser.parse_args()
    unique_id = int(time.time())
    
    metric_results = copy.deepcopy(args)

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
        #torch.cuda.manual_seed(random_seed) 
        #torch.cuda.manual_seed_all(random_seed) # if use multi-GPU 
        #torch.backends.cudnn.deterministic = True 
        #torch.backends.cudnn.benchmark = False 
        np.random.seed(random_seed) 
        random.seed(random_seed)
        
    args.out_root = os.path.join(args.out, args.target)
    if not os.path.exists(args.out_root):
        os.makedirs(args.out_root)
        
    ## initialization
    noise = get_noise(args)
    
    ## gen source image
    #source_imgs = os.path.join(args.out_root, "source_imgs")
    #generate_imgs_from_pt(args.source_ckpt, source_imgs, noise, False, args)
    #print("Source img generated")

    ## gen target training image
    train_imgs = os.path.join(args.out_root, "train_imgs")
    generate_imgs_from_ds(args.train_imgs, train_imgs, args)
    
    ## gen sample image
    sample_noise = get_sample_noise(args)
    if sample_noise != None:
        sample_imgs = os.path.join(args.out_root, "sample_imgs")
        generate_imgs_from_pt(args.model_ckpt, sample_imgs, sample_noise, args.parallel, args)
        print("sample img generated")
    
    ## gen model image
    model_imgs = os.path.join(args.out_root, "model_imgs")
    generate_imgs_from_pt(args.model_ckpt, model_imgs, noise, args.parallel, args)
    print("Model img generated")

    ## clurster image for LPIPS
    cLPIPS = os.path.join(args.out_root, "cLPIPS")
    LPIPS_mean, LPIPS_std = make_cluster(train_imgs, model_imgs, cLPIPS, args)
    print("LPIPS Result : ", LPIPS_mean, LPIPS_std)
    metric_results.cLPIPS = LPIPS_mean
    
    ## FID
    if args.test_imgs != None:
        #FID_w_Target = fid_score.calculate_fid_given_paths([args.test_imgs, model_imgs],batch_size=8,cuda=args.device,dims=2048)
        #print("FID Result : ", FID_w_Target)
        #metric_results.FID = FID_w_Target
        metric_results.fidelity = kid_score(args.test_imgs, model_imgs, args)
        print(metric_results.fidelity)
    else :
        batch_size = args.n_sample if args.n_sample < 100 else 100
        IS_mean, IS_std = inception_score(model_imgs, batch_size=batch_size, resize=False, splits=10)
        print("IS Result : ", IS_mean, IS_std)
        metric_results.IS = IS_mean

    ## Style Distance from source
    # style_dist, content_dist = distance_from_source(source_imgs, model_imgs, args)
    #metric_results.style_dist = style_dist
    #metric_results.content_dist = content_dist

    ## Save results
    log_path = "_result_%s_%d_%d.json" % (args.postfix, args.n_sample, unique_id)
    full_path = os.path.join(args.out_root, log_path)
    with open(full_path, 'w') as f:
        json.dump(metric_results.__dict__, f, indent=2)

    if os.path.exists("../metric_results"):
        _to = os.path.join("../metric_results", log_path)
        os.system("cp %s %s" %(full_path, _to))        
    