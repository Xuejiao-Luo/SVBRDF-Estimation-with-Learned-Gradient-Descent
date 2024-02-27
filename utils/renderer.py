import torch
import math
from utils.utils import normalize_length

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
    finalVec = torch.cat((x, y, z), dim=-1)
    
    return finalVec


def generate_normalized_random_direction_allpixels(batchSize, hight, width,lowEps=0.01, highEps=0.05):
    r1 = (1.0 - highEps - lowEps) * torch.rand([batchSize, 1, hight, width], dtype=torch.float32) + lowEps
    r2 = torch.rand([batchSize, 1, hight, width], dtype=torch.float32)
    r = torch.sqrt(r1)
    phi = 2 * math.pi * r2

    x = r * torch.cos(phi)
    y = r * torch.sin(phi)
    z = torch.sqrt(1.0 - torch.square(r))
    finalVec = torch.cat((x, y, z), dim=1)

    return finalVec

def generate_distance(batchSize, device):
    gaussian = torch.empty([batchSize, 1], dtype=torch.float32).normal_(mean=0.5,std=0.75).to(device)  # parameters chosen empirically to have a nice distance from a -1;1 surface.
    return (torch.exp(gaussian))

def diffuseRendering(batchSize, targets, outputs, device, args):    
    currentViewPos = generate_normalized_random_direction(batchSize).to(device)
    currentLightPos = generate_normalized_random_direction(batchSize).to(device)
    
    wi = currentLightPos
    wi = torch.unsqueeze(wi, dim=1)
    wi = torch.unsqueeze(wi, dim=1)
    
    wo = currentViewPos
    wo = torch.unsqueeze(wo, dim=1)
    wo = torch.unsqueeze(wo, dim=1)

    renderedDiffuse = render(targets,wi,wo, args)   
    renderedDiffuseOutputs = render(outputs,wi,wo, args )
    return [renderedDiffuse, renderedDiffuseOutputs]

def specularRendering(batchSize, surfaceArray, targets, outputs, device, args):    

    currentViewDir = generate_normalized_random_direction(batchSize).to(device)
    currentLightDir = currentViewDir * torch.unsqueeze(torch.Tensor([-1.0, -1.0, 1.0]), dim = 0).to(device)
    rand_val = 2.0 * torch.rand([batchSize, 2], dtype=torch.float32).to(device) - 1.0
    currentShift = torch.cat([rand_val, torch.zeros([batchSize, 1], dtype=torch.float32).to(device) + 0.0001], dim=-1)
    
    currentViewPos = currentViewDir * generate_distance(batchSize, device) + currentShift
    currentLightPos = currentLightDir * generate_distance(batchSize, device) + currentShift
    
    currentViewPos = torch.unsqueeze(currentViewPos, dim=1)
    currentViewPos = torch.unsqueeze(currentViewPos, dim=1)

    currentLightPos = torch.unsqueeze(currentLightPos, dim=1)
    currentLightPos = torch.unsqueeze(currentLightPos, dim=1)

    wo = currentViewPos - surfaceArray
    wi = currentLightPos - surfaceArray

    renderedSpecular = render(targets,wi,wo, args, includeDiffuse = args.includeDiffuse)
    renderedSpecularOutputs = render(outputs,wi,wo, args, includeDiffuse = args.includeDiffuse)#tf_Render_Optis(outputs,wi,wo, includeDiffuse = a.includeDiffuse)
    return [renderedSpecular, renderedSpecularOutputs]


def dotProduct_brdfs(tensorA, tensorB):
    return torch.sum(tensorA*tensorB, dim=-1, keepdim=True)

def render_diffuse_Substance(diffuse, specular):
    return diffuse * (1.0 - specular) / math.pi

def render_D_GGX_Substance(roughness, NdotH):
    alpha = torch.square(roughness)
    underD = 1/torch.clamp((torch.square(NdotH) * (torch.square(alpha) - 1.0) + 1.0), min=0.001)
    return (torch.square(alpha * underD)/math.pi)
    
def lampAttenuation(distance):
    DISTANCE_ATTENUATION_MULT = 0.001
    return 1.0 / (1.0 + DISTANCE_ATTENUATION_MULT*torch.square(distance))


def render_F_GGX_Substance(specular, VdotH):
    sphg = torch.pow(2.0, ((-5.55473 * VdotH) - 6.98316) * VdotH)
    return specular + (1.0 - specular) * sphg
    
def render_G_GGX_Substance(roughness, NdotL, NdotV):
    return G1_Substance(NdotL, torch.square(roughness)/2) * G1_Substance(NdotV, torch.square(roughness)/2)
    
def G1_Substance(NdotW, k):
    return 1.0/torch.clamp((NdotW * (1.0 - k) + k), min=0.001)

def render(svbrdf, wi, wo, args, includeDiffuse = True):
    ''' svbdrf : (BatchSize, Width, Height, 4 * 3)
        wo : (BatchSize,1,1,3)
        wi : (BatchSize,1,1,3) '''
        
    wiNorm = normalize_length(wi)
    woNorm = normalize_length(wo)
    h = normalize_length(torch.add(wiNorm,woNorm) / 2.0)
    diffuse = torch.clamp(de_preprocess_f01(svbrdf[:,:,:,3:6]), 0.0, 1.0)
    normals = svbrdf[:,:,:,0:3]
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
    result = specular_rendered
    
    if includeDiffuse:
        result = result + diffuse_rendered
    
    lampIntensity = 1.0
    lampFactor = lampIntensity * math.pi
    
    result = result * lampFactor

    result = result * torch.clamp(NdotL, min=0.0) / torch.unsqueeze(torch.clamp(wiNorm[:,:,:,2], min=0.001), dim=-1)
    return [result, D_rendered, G_rendered, F_rendered, diffuse_rendered, diffuse]