import torch
import matplotlib.pyplot as plt
from pdb import set_trace as bp

# from utils.renderer import normalize_length


def plot_img_and_mask(img, mask):
    classes = mask.shape[0] if len(mask.shape) > 2 else 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    if classes > 1:
        for i in range(classes):
            ax[i + 1].set_title(f'Output mask (class {i + 1})')
            ax[i + 1].imshow(mask[:, :, i])
    else:
        ax[1].set_title(f'Output mask')
        ax[1].imshow(mask)
    plt.xticks([]), plt.yticks([])
    plt.show()

def normalize(image):
    '''[0, 255] => [0, 1]'''        
    return image/255

def de_normalize(image):
    '''[0, 1] => [0, 255]'''        
    return image*255

def preprocess(image):
    '''[0, 1] => [-1, 1]'''        
    return image * 2 - 1

def de_preprocess(image):
    '''[-1, 1] => [0, 1]'''
    return (image + 1) / 2

def log_transform(image, delta):
    '''flatten the dynamic range by transforming the input image into logarithmic space'''    
    return (torch.log(torch.add(image, delta)) - torch.log(delta)) / (torch.log(delta * 101)-torch.log(delta))

def de_log_transform(image, delta):
    '''reverse the process: flatten the dynamic range by transforming the input image into logarithmic space'''
    return torch.exp(image * (torch.log(delta * 101)-torch.log(delta)) + torch.log(delta)) - delta 


def output_process(outputs):
    outputs = torch.permute(outputs, (0, 2, 3, 1))
    # partialOutputedNormals = outputs[:,:,:,0:2]
    # outputedDiffuse = outputs[:,:,:,2:5]
    # outputedRoughness = outputs[:,:,:,5]
    # outputedSpecular = outputs[:,:,:,6:9]
    partialOutputedNormals = outputs[:, :, :, 3:5]
    outputedDiffuse = outputs[:, :, :, 0:3]
    outputedRoughness = outputs[:, :, :, 5]
    outputedSpecular = outputs[:, :, :, 6:9]

    old_normal = False
    if old_normal:
        tmpNormals = torch.ones_like(partialOutputedNormals[..., 0]).unsqueeze(dim=-1)
        normNormals = normalize_length(torch.cat([partialOutputedNormals, tmpNormals], dim=-1))
    else:
        eps = 1e-6
        normal_xy = (partialOutputedNormals[...,0] ** 2 + partialOutputedNormals[...,1] ** 2).clamp(min=0, max=1 - eps)
        normal_z = (1 - normal_xy).sqrt().unsqueeze(-1)
        normNormals = normalize_length(torch.cat([partialOutputedNormals, normal_z], dim=-1))

    outputedRoughnessExpanded = torch.unsqueeze(outputedRoughness, dim=-1)
    outputs = torch.cat(
        [normNormals, outputedDiffuse, outputedRoughnessExpanded, outputedRoughnessExpanded, outputedRoughnessExpanded,
         outputedSpecular], dim=-1)

    return outputs


def wiwo_process(wi, wo, rendertype):
    assert rendertype=='DiffuseImg' or rendertype=='SpecularImg', 'wrong rendering type, should have been either DiffuseImg or SpecularImg'

    wi = wi.permute(0, 2, 3, 1)
    wo = wo.permute(0, 2, 3, 1)

    if rendertype=='DiffuseImg':
        wi=torch.mean(wi, [1,2], keepdim=True)
        wo=torch.mean(wo, [1,2], keepdim=True)

    return wi, wo

def pad_w(image, w, n = 3):
    # image_padded = image
    image_padded = torch.zeros_like(image)
    
    # image_padded[:,:,int(w/n)*(n-1): int(w/n)*n -1, :] = image[:,:,0:int(w/n)*1 -1 ,:]
    # for k in range(1,n):
    #     image_padded[:,:,int(w/n)*k: int(w/n)*(k+1) -1, :] = image[:,:,int(w/n)*(k+1): int(w/n)*(k+2) -1, :]

    image_padded[:,:,0:int(w/n)*1,:] = image[:,:,int(w/n)*1: int(w/n)*2,:]
    image_padded[:,:,int(w/n)*1: int(w/n)*2, :] = image[:,:,int(w/n)*2: int(w/n)*3, :]
    image_padded[:,:,int(w/n)*2: w, :] = image[:,:,0:w-int(w/n)*2,:]
    
    return image_padded

def pad_h(image, h, n = 3):
    # image_padded = image
    image_padded = torch.zeros_like(image)
    
    # image_padded[:,:,int(w/n)*(n-1): int(w/n)*n -1, :] = image[:,:,0:int(w/n)*1 -1 ,:]
    # for k in range(1,n):
    #     image_padded[:,:,int(w/n)*k: int(w/n)*(k+1) -1, :] = image[:,:,int(w/n)*(k+1): int(w/n)*(k+2) -1, :]

    image_padded[:,:,:,0:int(h/n)*1] = image[:,:,:,int(h/n)*1: int(h/n)*2]
    image_padded[:,:,:,int(h/n)*1: int(h/n)*2] = image[:,:,:,int(h/n)*2: int(h/n)*3]
    image_padded[:,:,:,int(h/n)*2: h] = image[:,:,:,0:h-int(h/n)*2]
    
    return image_padded

def normalize_length(tensor):
    '''Normalizes a tensor throughout the Channels dimension (BatchSize, Width, Height, Channels)
       Keeps 4th dimension to 1. Output will be (BatchSize, Width, Height, 1).'''
    
    length = torch.sqrt(torch.sum(torch.square(tensor), dim = -1, keepdim=True))
    return torch.div(tensor, length)


