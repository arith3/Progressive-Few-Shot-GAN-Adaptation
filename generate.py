import argparse
import random
import torch
import torch.nn as nn
from torchvision import utils
from model import Generator
from tqdm import tqdm
import sys
import os
from model import Generator, Projection_module, Projection_module_church

def generate_gif(args, g_list, device, mean_latent):
    with torch.no_grad():
        for i in range(len(g_list)):
            g_test = g_list[i]
            g_test.eval()

        z_set = torch.load('noise.pt')

        n_steps = args.n_steps
        step = float(1)/n_steps
        n_paths = z_set.size(0)
        for t in tqdm(range(n_paths)):
            if t != (n_paths - 1):
                z1, z2 = torch.unsqueeze(
                    z_set[t], 0), torch.unsqueeze(z_set[t+1], 0)
            else:
                z1, z2 = torch.unsqueeze(
                    z_set[t], 0), torch.unsqueeze(z_set[0], 0)

            for i in range(n_steps):
                alpha = step*i
                z = z2*alpha + (1-alpha)*z1
                for m in range(len(g_list)):
                    sample, _ = g_list[m]([z], truncation=args.truncation,
                                      truncation_latent=mean_latent, randomize_noise=False)
                    if m == 0:
                        tot_img = sample
                    else:
                        tot_img = torch.cat([tot_img, sample], dim = 3)

                utils.save_image(
                    tot_img,
                    f'./outputs/traversals/sample%d.png' % ((t*n_steps) + i),
                    normalize=True,
                    range=(-1, 1),
                )


def generate_imgs(args, g_test, device, mean_latent):

    with torch.no_grad():
        g_test.eval()
        if args.load_noise:
            sample_z = torch.load(args.load_noise)
        else:
            sample_z = torch.randn(args.n_sample, args.latent, device=device)

        sample, _ = g_test([sample_z], truncation=args.truncation, truncation_latent=mean_latent, input_is_latent=False, randomize_noise=False)
        if i == 0:
            tot_img = sample
        else:
            tot_img = torch.cat([tot_img, sample], dim = 0)

    utils.save_image(
     tot_img,
     f'test_sample/sample.png',
     nrow=5,
     normalize=True,
     range=(-1, 1),
     )

def generate_imgs_4IS(args, g_target, Proj_module):
    
    with torch.no_grad():
        
        batch = 50
        sample_z = torch.randn(args.IS_sample, args.latent).cuda()
        step = int(args.IS_sample / batch)
        for i in tqdm(range(int(args.IS_sample / batch))):
            if args.latent_dir != "":
                w = [g_target.module.style(sample_z[i*batch: (i+1)*batch])]
                w = [Proj_module.modulate(item) for item in w]
                sample_t, _= g_target(w, input_is_latent=True, randomize_noise=False)
            else:
                w = sample_z[i*batch: (i+1)*batch]
                sample_t, _= g_target([w], input_is_latent=False, randomize_noise=False)
            for (num, img) in enumerate(sample_t):
                utils.save_image(
                    img,
                    f'%s/img%d.png' % (args.save_target, (i * batch) + num) ,
                    normalize=True,
                    range=(-1, 1),
                )
        #print(i)

if __name__ == '__main__':
    device = 'cuda'

    parser = argparse.ArgumentParser()

    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--n_sample', type=int, default=25, help='number of fake images to be sampled')
    parser.add_argument('--IS_sample', type=int, default=10000, help='number of fake images to be sampled for IS')
    parser.add_argument('--n_steps', type=int, default=40, help="determines the granualarity of interpolation")
    parser.add_argument('--truncation', type=float, default=1)
    parser.add_argument('--truncation_mean', type=int, default=4096)
    parser.add_argument('--ckpt_source', type=str, default=None)
    parser.add_argument('--ckpt_target', type=str, default=None)
    parser.add_argument('--ckpt_target2', type=str, default=None)
    parser.add_argument('--mode', type=str, default='viz_imgs')
    parser.add_argument('--name', type=str, default='')
    parser.add_argument('--load_noise', type=str, default=None)
    parser.add_argument('--channel_multiplier', type=int, default=2)
    parser.add_argument('--latent_dir', type=str, default="")
    parser.add_argument('--target', type=str, default='VanGogh', help='target domain')
    parser.add_argument('--task', type=int, default=10)
    parser.add_argument('--save_target', type=str, default=None)
    
    #torch.manual_seed(10)
    #random.seed(10)
    args = parser.parse_args()
    args.latent = 512
    args.n_mlp = 8
    args.exp_name = args.target

    g_list = []
    
    # loading target model if available
    if args.ckpt_target is not None:
        g_target = Generator(
            args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
        ).to(device)
        g_target = nn.parallel.DataParallel(g_target)
        checkpoint = torch.load(args.ckpt_target)
        g_target.load_state_dict(checkpoint['g_ema'], strict=False)
        g_list.append(g_target)

    # loading source model if available
    if args.ckpt_source is not None:
        g_source = Generator(
            args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
        ).to(device)
        checkpoint = torch.load(args.ckpt_source)
        g_source.load_state_dict(checkpoint['g_ema'], strict=False)
        g_list.append(g_source)
    
    # loading target model if available
    if args.ckpt_target2 is not None:
        g_target2 = Generator(
            args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
        ).to(device)
        g_target2 = nn.parallel.DataParallel(g_target2)
        checkpoint = torch.load(args.ckpt_target2)
        g_target2.load_state_dict(checkpoint['g_ema'], strict=False)
        g_list.append(g_target2)
        
    if args.truncation < 1:
        with torch.no_grad():
            mean_latent = g_source.mean_latent(args.truncation_mean)
    else:
        mean_latent = None
    if args.mode == 'viz_imgs':
        generate_imgs(args, g_list, device, mean_latent)
    elif args.mode == 'interpolate':
        generate_gif(args, g_list, device, mean_latent)
    elif args.mode == 'is':
        if args.save_target == None:
            args.save_target = os.path.join("./outputs/metric/tmp/", args.name)
        else :
            args.save_target = os.path.join(args.save_target, args.name)
            
        if not os.path.exists(args.save_target):
            os.makedirs(args.save_target)
        Proj_module = None
        if args.latent_dir != "":
            Proj_module = Projection_module(args)
        generate_imgs_4IS(args, g_target, Proj_module)

