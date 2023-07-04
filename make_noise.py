import argparse
import easydict
import math
import random
import os
import sys
import numpy as np
import torch

def gen_noise(args, device):
    noise = torch.randn(args.n_sample, args.latent, device=device)
    print("noise generated : ", noise.shape)
    return noise

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser()

    parser.add_argument("out", type=str)

    parser.add_argument("--size", type=int, default=256) # image size
    parser.add_argument("--latent", type=int, default=512)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n_sample", type=int, default=100)
    parser.add_argument("--batch", type=int, default=32)

    args = parser.parse_args()
    

    if args.seed != None:
        random_seed = args.seed
        torch.manual_seed(random_seed) 
        np.random.seed(random_seed) 
        random.seed(random_seed)

    sample_z = torch.randn(args.n_sample, args.latent, device=device)

    torch.save(sample_z, args.out)