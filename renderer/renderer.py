# -*- coding: utf-8 -*-
import os
import torch
import math
import h5py
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pdb import set_trace as bp  
from torch.utils.data import DataLoader

from data_loading_brdf import BasicDataset, BasicDataset_maps_highlightaware
from utils import normalize, de_normalize, preprocess, de_preprocess, output_process, log_transform, swin
from rendererMG import renderTex


def de_preprocess_f01(image):
    '''[-1, 1] => [0, 1]'''
    return (image + 1) / 2

def generate_normalized_random_direction(batchSize, lowEps = 0.01, highEps =0.05):
        
    r1 = (1.0 - highEps - lowEps) * torch.rand([batchSize, 1], dtype=torch.float32) + lowEps
    r2 = torch.rand([batchSize, 1], dtype=torch.float32)
    r = torch.sqrt(r1)
    phi = 2 * math.pi * r2
       
    x = r * torch.cos(phi)
    y = r * torch.sin(phi)
    z =  torch.sqrt(1.0 -  torch.square(r))
    finalVec = torch.cat((x, y, z), dim=-1) #Dimension here is [batchSize, 3]
    
    return finalVec

def generate_distance(batchSize, device):
    # gaussian = torch.empty([batchSize, 1], dtype=torch.float32).normal_(mean=0.5,std=0.75).to(device)  # parameters chosen empirically to have a nice distance from a -1;1 surface.
    gaussian = torch.empty([batchSize, 1], dtype=torch.float32).normal_(mean=3.5,std=0.75).to(device)  # parameters chosen empirically to have a nice distance from a -1;1 surface.
    gaussian[gaussian<0.0]=0.0
    return (torch.exp(gaussian))


def normalize_length(tensor):
    '''Normalizes a tensor throughout the Channels dimension (BatchSize, Width, Height, Channels)
       Keeps 4th dimension to 1. Output will be (BatchSize, Width, Height, 1).'''
    
    length = torch.sqrt(torch.sum(torch.square(tensor), dim = -1, keepdim=True))
    return torch.div(tensor, length)
    

def dotProduct_brdfs(tensorA, tensorB): 
    '''Computes the dot product between 2 tensors (BatchSize, Width, Height, Channels)
        Keeps 4th dimension to 1. Output will be (BatchSize, Width, Height, 1).'''       
    return torch.sum(tensorA*tensorB, dim=-1, keepdim=True)

def render_diffuse_Substance(diffuse, specular):
    return diffuse * (1.0 - specular) / math.pi

def render_D_GGX_Substance(roughness, NdotH):
    alpha = torch.square(roughness)
    underD = 1/torch.clamp((torch.square(NdotH) * (torch.square(alpha) - 1.0) + 1.0), min=0.001)
    return (torch.square(alpha * underD)/math.pi)
    
def lampAttenuation(distance):
    DISTANCE_ATTENUATION_MULT = 0.001
    return 1.0 / (1.0 + DISTANCE_ATTENUATION_MULT*torch.square(distance));


def render_F_GGX_Substance(specular, VdotH):
    sphg = torch.pow(2.0, ((-5.55473 * VdotH) - 6.98316) * VdotH);
    return specular + (1.0 - specular) * sphg
    
def render_G_GGX_Substance(roughness, NdotL, NdotV):
    return G1_Substance(NdotL, torch.square(roughness)/2) * G1_Substance(NdotV, torch.square(roughness)/2)
    
def G1_Substance(NdotW, k):
    return 1.0/torch.clamp((NdotW * (1.0 - k) + k), min=0.001)

def render(svbrdf, wi, wo):
    ''' svbdrf : (BatchSize, Width, Height, 4 * 3)
        wo : (BatchSize,1,1,3)
        wi : (BatchSize,1,1,3) '''
        
    wiNorm = normalize_length(wi)
    woNorm = normalize_length(wo)
    h = normalize_length(torch.add(wiNorm,woNorm) / 2.0)
    diffuse = torch.clamp(de_preprocess_f01(svbrdf[:,:,:,3:6]), 0.0, 1.0) #[0, 1]
    normals = svbrdf[:,:,:,0:3]  #[-1, 1]
    specular = torch.clamp(de_preprocess_f01(svbrdf[:,:,:,9:12]), 0.0, 1.0)
    roughness = torch.clamp(de_preprocess_f01(svbrdf[:,:,:,6:9]), 0.0, 1.0)
    roughness = torch.clamp(roughness, min=0.001)
    NdotH = dotProduct_brdfs(normals, h)
    NdotL = dotProduct_brdfs(normals, wiNorm)
    NdotV = dotProduct_brdfs(normals, woNorm)
    VdotH = dotProduct_brdfs(woNorm, h)


    diffuse_rendered = render_diffuse_Substance(diffuse, specular)
    D_rendered = render_D_GGX_Substance(roughness, torch.clamp(NdotH, min=0.0))
    G_rendered = render_G_GGX_Substance(roughness, torch.clamp(NdotL, min=0.0), torch.clamp(NdotV, min=0.0))
    F_rendered = render_F_GGX_Substance(specular, torch.clamp(VdotH, min=0.0))
    
    specular_rendered = F_rendered * (G_rendered * D_rendered * 0.25)
    
    result = specular_rendered + diffuse_rendered
    
    lampIntensity = 1.0
    #lampDistance = torch.sqrt(torch.sum(torch.square(wi), dim = 3, keepdim=True))
    
    lampFactor = lampIntensity * math.pi # lampAttenuation(lampDistance) * lampIntensity * math.pi
    
    result = result * lampFactor

    result = result * torch.clamp(NdotL, min=0.0) / torch.unsqueeze(torch.clamp(wiNorm[:,:,:,2], min=0.001), dim=-1) # This division is to compensate the cosinus distribution of the intensity in the rendering
            
    return result


def diffuseRendering(batchSize, targets, device):    
    currentViewPos = generate_normalized_random_direction(batchSize).to(device)
    currentLightPos = generate_normalized_random_direction(batchSize).to(device)
    
    wi = currentLightPos
    wi = torch.unsqueeze(wi, dim=1)
    wi = torch.unsqueeze(wi, dim=1)
    
    wo = currentViewPos
    wo = torch.unsqueeze(wo, dim=1)
    wo = torch.unsqueeze(wo, dim=1)
    #[result, D_rendered, G_rendered, F_rendered, diffuse_rendered, specular_rendered]
    renderedDiffuse = render(targets,wi,wo)   
    
    return [renderedDiffuse, wi, wo]

def specularRendering(batchSize, surfaceArray, targets, device):    
    currentViewPos = torch.tensor([[0.0,0.0,16.0]]).to(device)
    currentLightPos = torch.tensor([[0.0,0.0,16.0]]).to(device)

    wo = currentViewPos - surfaceArray
    wi = currentLightPos - surfaceArray

    renderedSpecular = render(targets,wi,wo)           
    return [renderedSpecular, wi, wo]


if __name__ == '__main__':
    nbDiffuseRendering = 3
    nbSpecularRendering = 3    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load data
    dir_img = Path('./img_highlight_aware/map/')

    img_scale = 1.0

    highlight_flag = 'highlight' in str(dir_img)

    dataset = BasicDataset(dir_img, img_scale) if not highlight_flag else BasicDataset_maps_highlightaware(dir_img, img_scale)

    loader_args = dict(batch_size=1, num_workers=1, pin_memory=True)
    dataset_loader = DataLoader(dataset, shuffle=False, generator=torch.Generator().manual_seed(0), **loader_args)

    idx = 0
    for batch in dataset_loader:
        batch, img_idx = batch
        if img_idx%1==0:
            if idx%10000==0:
                print(idx)
            idx = idx + 1
            print(batch[0])

            if os.name == 'nt':
                fileseparate = '\\'
            else:
                fileseparate = '/'
            fn_tex = str(dir_img)+fileseparate+batch[0]+'.png'

            # save rendered image using the new render: renderTex
            saveRenderedImage = True
            if saveRenderedImage:
                folder_save = str(dir_img) + '_rendered'
                if not os.path.exists(folder_save):
                    os.makedirs(folder_save)

                nr_scales = 1
                np.random.seed(1)
                lp = np.array((0,0,250), dtype=int)
                cp = np.array((0,0,250), dtype=int)
                np.random.seed(2)
                qx = np.random.uniform(low=-100, high=100, size=nr_scales)
                qy = np.random.uniform(low=-100, high=100, size=nr_scales)

                base_lightIntensity = 60000
                np.random.seed(0)
                lightScaling = 20 * np.random.uniform(low=0.0, high=1.0, size=nr_scales)
                for i in range(nr_scales):
                    height = 250
                    lightIntensity = int(329288)
                    L = np.array((lightIntensity,), dtype=int)  
                    if os.name == 'nt':
                        fileseparate = '\\'
                    else:
                        fileseparate = '/'
                    fn_tex_save = folder_save + fileseparate + batch[0] + '_ligInt_'+str(lightIntensity) +'.png'

                    renderTex(fn_tex, lp, cp, L, fn_tex_save, highlight_flag)