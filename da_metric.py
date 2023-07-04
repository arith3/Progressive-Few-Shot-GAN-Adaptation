import argparse
import random
import torch
import torch.nn as nn
from torch.nn import functional as F

from torchvision import utils
from model import Generator, Extra, Projection_module, Projection_module_church
from model import Patch_Discriminator as Discriminator  # , Projection_head

#from tqdm.notebook import tqdm
from tqdm import tqdm
import sys
import os
import PIL
from PIL import Image
from PIL import ImageFile
#from matplotlib import pyplot as plt
#import matplotlib
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import lpips
from pytorch_fid import fid_score, inception
import cv2
from skimage.metrics import structural_similarity as compare_ssim
import imutils
import shutil
import imageio
import os
from dataset import MultiResolutionDataset

import easydict
import json

GENERATING_STEP = 32

def search(dirname):
    result = []
    filenames = list(filter(lambda x: not x.startswith('.'), os.listdir(dirname)))
    for filename in filenames:
        full_filename = os.path.join(dirname, filename)
        result.append(full_filename)
    return result
        
class GenerateFromGAN():
    def __init__(self):
        super().__init__()
        print("init generate methods!!")
        torch.manual_seed(99)
        random.seed(99)
        self.g_ema = None
        self.d = None
        
    def make_noise(self, args, device, skip_number=0):
        if args.load_noise:
            noise = torch.load(args.load_noise)
            if len(noise.shape) > 3:
                noise = torch.load(args.load_noise)[:,0:args.n_sample+skip_number].to(device)
            else:
                noise = torch.load(args.load_noise)[0:args.n_sample+skip_number].to(device)
        else:
            noise = torch.randn(args.n_sample+skip_number, args.latent, device=device)
        return noise
    
    def load_model(self, args, device, targetCkpt=None, parallel=False):
        if targetCkpt == None:
            targetCkpt = args.ckpt
            
        # loading source model if available
        if args.ckpt is not None:
            model = Generator(
                args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
            ).to(device)
            checkpoint = torch.load(targetCkpt)
            if parallel == True:
                model = nn.parallel.DataParallel(model)
                model.load_state_dict(checkpoint['g_ema'], strict=False)
            else:
                model.load_state_dict(checkpoint['g_ema'], strict=False)
                model = nn.parallel.DataParallel(model)
            model.eval()
            
            self.g_ema = model
            return model
        
    def reset_outfolder(self, path):
        if os.path.exists(path):
            try:
                shutil.rmtree(path)
            except OSError as e:
                print ("Error: %s - %s." % (e.filename, e.strerror))
        os.makedirs(path)
    
    def requires_grad(self, model, flag=True):
        for name, p in model.named_parameters():
            p.requires_grad = flag

    def get_image(self, args, device):
        args.latent = 512
        args.n_mlp = 8
        z = torch.randn(args.n_sample, args.latent, device=device)
        if self.g_ema == None:
            self.g_ema = self.load_model(args, device, args.ckpt, args.parallel)
        
        if args.truncation < 1:
            with torch.no_grad():
                mean_latent = model.mean_latent(args.truncation_mean)
        else:
            mean_latent = None
        with torch.no_grad():
            self.g_ema.eval()
            self.requires_grad(self.g_ema, False)
            sample, _ = self.g_ema([z])
            
        return sample, z
    
    def discrimination(self, args, imgs, device):
        if self.d == None:
            self.d = Discriminator(
                args.size, channel_multiplier=args.channel_multiplier
            ).to(device)
            checkpoint = torch.load(args.ckpt)
            self.extra = Extra().to(device)
            if args.parallel == True:
                self.d = nn.parallel.DataParallel(self.d)
                self.d.load_state_dict(checkpoint['d'], strict=False)
                self.extra = nn.parallel.DataParallel(self.extra)
            self.d.eval()
        lowp, highp = 0, 1
        which = 0
        pred = self.d(
            imgs, extra=self.extra, flag=which, p_ind=np.random.randint(lowp, highp), real=False)
        return pred
        
    def _genrate_image_by_dataset(self, args):
        self.reset_outfolder(args.out)
        transform = transforms.Compose([])
        dataset = MultiResolutionDataset(args.ckpt, transform, args.size)
        for index in range(len(dataset)):
            name = args.out + (f'/sample%03d.png' % (index))
            dataset[index].save(name, "png")
        del dataset
    
    def _genrate_image_by_pt(self, args, device):
        args.latent = 512
        args.n_mlp = 8
        noise = self.make_noise(args, device)
        model = self.load_model(args, device, args.ckpt, args.parallel)
        self.reset_outfolder(args.out)

        if args.latent_dir != None:
            args.task = 10
            Proj_module = Projection_module(args)
            print("!!!generation on w space!!!")
            
        if args.truncation < 1:
            with torch.no_grad():
                mean_latent = model.mean_latent(args.truncation_mean)
        else:
            mean_latent = None

        with torch.no_grad():
            model.eval()
            self.requires_grad(model, False)
            
            for start in tqdm(range(0,args.n_sample, GENERATING_STEP), leave=False):
                if start + GENERATING_STEP <= args.n_sample:
                    end = start + GENERATING_STEP
                    if args.latent_dir != None:
                        w = [model.module.style(noise[start:end])]
                        w = [Proj_module.modulate(item) for item in w]
                        sample, _ = model(w, input_is_latent=True, randomize_noise=False)
                    else:
                        sample, _ = model([noise[start:end]])
                    for index in range(sample.shape[0]):
                        utils.save_image(
                            sample[index],
                            args.out + (f'/sample%03d.png' % (start+index)),
                            nrow=5,
                            normalize=True,
                            range=(-1, 1),
                            )
                else:
                    for index in range(start, args.n_sample):
                        # noise가 tuple인 경우 고려 필요. 
                        
                        if len(noise.shape) > 2:
                            z = noise[:,index,:].unsqueeze(1).unbind(0)
                            print(z.shape)
                        else:
                            z = noise[index,:].unsqueeze(0)
                            
                        if args.latent_dir != None:
                            w = [model.module.style(z)]
                            w = [Proj_module.modulate(item) for item in w]
                            sample, _ = model(w, input_is_latent=True, randomize_noise=False)
                        else:
                            sample, _ = model([z])
                        utils.save_image(
                            sample,
                            args.out + (f'/sample%03d.png' % index),
                            nrow=5,
                            normalize=True,
                            range=(-1, 1),
                            )
        del model
        del noise
        
    def generate_image(self, args, device):
        assert args.ckpt != None
        
        if not args.ckpt.endswith("pt"):
            self._genrate_image_by_dataset(args)
        else:
            self._genrate_image_by_pt(args, device)
        
        
    def generate_gif(self, args, device):
        args.latent = 512
        args.n_mlp = 8
        noise = self.make_noise(args, device, args.skip_number)
        #self.reset_outfolder(args.out)
    
        samples = None
        for target in args.ckpt:
            model = self.load_model(args, device, target.ckpt, target.parallel)
            if args.truncation < 1:
                with torch.no_grad():
                    mean_latent = model.mean_latent(args.truncation_mean)
            else:
                mean_latent = None
                
            with torch.no_grad():
                step = float(1)/args.n_steps
                images = None
                for n in tqdm(range(args.n_sample)):
                    n = n+ args.skip_number
                    z1, z2 = torch.unsqueeze(noise[n], 0), torch.unsqueeze(noise[(n+1)%args.n_sample], 0)
                    latents = None
                    for i in range(args.n_steps):
                        alpha = step*i
                        z = z2*alpha + (1-alpha)*z1
                        latents = z if latents is None else torch.cat((latents, z), dim=0)
                    image, _ = model([latents], truncation=args.truncation, 
                                  truncation_latent=mean_latent, 
                                  input_is_latent=False, randomize_noise=False)
                    images = image if images is None else torch.cat((images, image), dim=3)
                samples = images if samples is None else torch.cat((samples, images), dim=2)
            del model
        samples = np.transpose(samples.cpu().squeeze(0), (0,2,3,1)).numpy()
        imageio.mimsave(args.out, samples, duration = 0.2)
        del noise        
            
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

class MetricBetwenGANs():
    def __init__(self, use_gpu=True):
        super().__init__()
        print("init metric methods!!")
        self.loss_fn_vgg = lpips.PerceptualLoss(
            model="net-lin", net="vgg", use_gpu=use_gpu
        )
        self.model_vgg = model_vgg = models.vgg19(pretrained=True).features.to("cuda" if use_gpu else "cpu").eval()
        #self.model_vgg = nn.parallel.DataParallel(self.model_vgg)
        for param in model_vgg.parameters():
            param.requires_grad_(False)
        
        self.transform = transforms.Compose(
            [
                #transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
            ]
        )
        self.content_result = {}
        
    def fid(self, source, target, device):
        stats = fid_score.calculate_fid_given_paths([source, target], batch_size=8,cuda=device,dims=2048)
        #print("FID:", paths, stats)
        return stats

    def get_path(self, path):
        #result = sorted(search(path))
        result = search(path)
        #random.shuffle(result)
        return result
    
    def lpips_style(self, source, target, device):
        sim = nn.CosineSimilarity()
        style_result = []
        content_result = []
        
        source_path = self.get_path(source)
        target_path = self.get_path(target)

        size = np.min([len(source_path), len(target_path)])
        
        for i in tqdm(range(size), leave=False):
            name = 'sample%03d.png' % i
            source_img = torch.unsqueeze(self.transform(PIL.Image.open(source_path[i])),0).to(device)
            target_img = torch.unsqueeze(self.transform(PIL.Image.open(target_path[i])),0).to(device)
            source_feat = get_features(source_img, self.model_vgg, feature_layers)
            target_feat = get_features(target_img, self.model_vgg, feature_layers)
            
            style_loss = get_style_loss(source_feat, target_feat, style_layers_dict)
            content_loss = get_content_loss(source_feat, target_feat, content_layer)
            
            style_result.append(style_loss.detach().cpu().item())
            content_result.append(content_loss.detach().cpu().item())
            
            del source_feat, target_feat, style_loss, content_loss
        self.content_result[target] = np.array(content_result).mean()
        return np.array(style_result).mean()
    
    def lpips_content(self, source, target, device):
        return self.content_result[target]
    
    def lpips(self, source, target, device):
        result = []
        
        source_path = self.get_path(source)
        target_path = self.get_path(target)

        size = np.min([len(source_path), len(target_path)])
        
        for i in tqdm(range(size), leave=False):
            source_img = torch.unsqueeze(self.transform(PIL.Image.open(source_path[i])),0).to(device)
            target_img = torch.unsqueeze(self.transform(PIL.Image.open(target_path[i])),0).to(device)
            dist = self.loss_fn_vgg(source_img, target_img).squeeze().mean().detach().cpu().item()
            result.append(dist)
        return np.array(result).mean()
    
    def lpips_wReal(self, real, target):
        result = []
        
        real_set = MultiResolutionDataset(real, self.transform, 256)
        #target_path = sorted(search(target))
        target_path = search(target)
        random.shuffle(target_path)
        
        size = np.min([len(real_set), len(target_path)])
        for i in tqdm(range(size)):
            source_img = torch.unsqueeze(real_set[i],0)
            target_img = torch.unsqueeze(self.transform(PIL.Image.open(target_path[i])),0)
            dist = self.loss_fn_vgg(source_img, target_img).squeeze().mean().detach().cpu().numpy().item()
            result.append(dist)
        return np.array(result).mean()
    
    def ssim(self, source, target, device):
        result = []
        
        source_path = self.get_path(source)
        target_path = self.get_path(target)

        size = np.min([len(source_path), len(target_path)])
        
        for i in tqdm(range(size), leave=False):
            source_img = cv2.imread(source_path[i])
            target_img = cv2.imread(target_path[i])
            source_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2GRAY)
            target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
    
            (score, diff) = compare_ssim(source_img, target_img, full=True)
            result.append(score)
        return np.array(result).mean()
    
    def cosineSimilarity(self, source, target, device):
        sim = nn.CosineSimilarity()
        result = []
        source_path = self.get_path(source)
        target_path = self.get_path(target)

        size = np.min([len(source_path), len(target_path)])
        for i in tqdm(range(size)):
            source_img = self.transform(cv2.imread(source_path[i]))
            target_img = self.transform(cv2.imread(target_path[i]))
            score = sim(source_img, target_img)
            result.append(score.mean())
        return np.array(result).mean()

class MetricProcess():
    def __init__(self, use_gpu):
        super().__init__()
        self.generator = GenerateFromGAN()
        self.metric = MetricBetwenGANs(use_gpu=use_gpu)
        self.metric_reports = easydict.EasyDict({})
        self.metric_func = {
            "fid":self.metric.fid,
            "lpips":self.metric.lpips,
            "inter_style":self.metric.lpips_style,
            "inter_content":self.metric.lpips_content,
            "intra_style":self.metric.lpips_style,
            "intra_content":self.metric.lpips_content,
            "ssim":self.metric.ssim,
            "cosineSimilarity":self.metric.cosineSimilarity,
        }
    def init_metric(self, args):
        self.metric_args = args
        self.device = torch.device(args.device)
        self._generate_images(args)
        
    def _generate_images(self, args):
        print("Generating test images....")
        generating_args = easydict.EasyDict({
            "size":args.size,
            "n_sample":args.n_sample,
            "truncation": args.truncation, 
            "truncation_mean": args.truncation_mean, 
            "load_noise": args.load_noise, 
            "channel_multiplier": args.channel_multiplier, 
            "parallel": False,
            "out": None,
            "ckpt": None,
            "latent_dir": args.latent_dir,
            "exp_name": args.exp_name,
        })
        for target in args.images:
            img_args = args.images[target]
            if "ckpt" in img_args.keys(): # manual ckpt
                img_args.out = generating_args.out = args.out_path + target
                generating_args.ckpt = img_args.ckpt
            else: 
                img_args.out = generating_args.out = args.out_path + target + "/" + args.ckpt_name
                generating_args.ckpt = args.ckpt_path + target + "/" + args.ckpt_name + ".pt"
                
            if not img_args.skip or not os.path.isdir(img_args.out):
                if "parallel" in img_args.keys(): # 없으면 Default False
                    generating_args.parallel = img_args.parallel
                else:
                    generating_args.parallel = False
                self.generator.generate_image(generating_args, self.device)
        print("Done generating...")
        
    def calc_metric(self, source, source_path, target, metric):
        target_labels = self.metric_args.metric_targets[target]
        if not target in self.metric_reports.keys():
            self.metric_reports[target] = easydict.EasyDict({})
        self.metric_reports[target][metric] = easydict.EasyDict({})
        
        #if metric != 'fid':
        #    target_labels = tqdm(target_labels)
            
        for label in target_labels:
            if source_path == None:
                result = self.metric_func[metric](self.metric_args.images[source].out,
                                                  self.metric_args.images[label].out, 
                                                  self.device)
            else:
                result = self.metric_func[metric](source_path,
                                                  self.metric_args.images[label].out, 
                                                  self.device)
            self.metric_reports[target][metric][label] = result
        return self.metric_reports[target][metric]
        
    def get_metric_report(self):
        return self.metric_reports
    
    def make_metric_report(self, source, source_path, target):
        ## FID Score (↓)
        fid_score = self.calc_metric(source, source_path, target, "fid")

        ## LPIPS (inter domain ↑)
        _ = self.calc_metric(source, source_path, target, "lpips")
        _ = self.calc_metric(source, source_path, target, "inter_style")
        _ = self.calc_metric(source, source_path, target, "inter_content")
        _ = self.calc_metric(target, source_path, target, "intra_style")
        _ = self.calc_metric(target, source_path, target, "intra_content")

        ## Structural SSIM
        ssim_result = self.calc_metric(source, source_path, target, "ssim")
        
        return self.metric_reports
    
def make_metric_result(args, ckpt, device):
    print("...")
    print("... calculating metric ...")
    target = args.data_path.split("/")[-1]
    metric_args = easydict.EasyDict({ 
        "source": args.source_key,
        "source_path": args.metric_source_path,
        "size": 256, 
        "n_sample": 1000, 
        "truncation": 1, 
        "truncation_mean": 4096,
        "load_noise": './data/ckpt/noise_metric_1000.pt', 
        "channel_multiplier": 2, 
        "device": device, 

        "out_path": './outputs/metric/images/',
        "ckpt_path": './checkpoints/',
        "ckpt_name": ckpt,
        "latent_dir": args.latent_dir,
        "exp_name": args.exp_name,
        "images":
        {
            args.source_key:{"ckpt":args.ckpt,"skip":True},
            target:{"ckpt":args.data_path,"skip":True},
            args.exp:{"parallel":True,"skip":False},
        },
        "metric_targets":
        {
            target:[args.exp],
        }
    })
    
    imsave_path = os.path.join('outputs/samples', args.exp)
    
    if not os.path.exists(imsave_path):
        os.makedirs(imsave_path)
            
    if args.metric_source_path == None:
        metric_args_path = imsave_path + "/_metric_args_" + metric_args.ckpt_name + ".json"
    else:
        metric_args_path = imsave_path + "/_metric_args_asSubset_" + metric_args.ckpt_name + ".json"
        
    with open(metric_args_path, 'w') as f:
        json.dump(metric_args.__dict__, f, indent=2)
    
    metric_proc = MetricProcess(device=="cuda")    
    metric_proc.init_metric(metric_args)
    metric_result = metric_proc.make_metric_report(args.source_key, args.metric_source_path, target)

    if args.metric_source_path == None:
        metric_result_path = imsave_path + "/_metric_result_" + metric_args.ckpt_name + ".json"
    else: 
        metric_result_path = imsave_path + "/_metric_asSubset_result_" + metric_args.ckpt_name + ".json"
        
    with open(metric_result_path, 'w') as f:
        json.dump(metric_result.__dict__, f, indent=2)

    del metric_proc
    print("... done metric ...")
    