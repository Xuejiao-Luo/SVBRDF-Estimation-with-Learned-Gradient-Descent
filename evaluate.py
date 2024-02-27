import torch
import wandb
from tqdm import tqdm

from utils.loss import loss_func
from utils.rendererMG import tex2map
from utils.rendererMG_util import gyApplyGamma
from utils.utils import de_normalize, preprocess, log_transform

def evaluate(net, dataloader, args, device, experiment):
    net.eval()
    num_val_batches = len(dataloader)
    loss = 0
    count = 0

    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        images = batch['image']
        brdfs = batch['brdfs']

        images = images.to(device=device, dtype=torch.float32)
        brdfs = brdfs.to(device=device, dtype=torch.float32)

        if args.correctGamma:
            images = torch.pow(images, 2.2)

        if args.useLog:
            delta = torch.tensor(0.01).to(device=device, dtype=torch.float32)                
            images_log = log_transform(images, delta)
        else:
            images_log = images

        eta = preprocess(images_log)

        with torch.no_grad():
        # predict the brdfs
            brdfs_pred = net(eta, images, args)
            brdfs_pred = brdfs_pred[1] # for evaluation, use only final output (so dismiss unet output)

        loss_results = loss_func(brdfs, brdfs_pred, images, args)
        loss = loss + loss_results[3]

        if count%1==0:
            reproduceImg = loss_results[0]
            rerenderedImg = loss_results[1]
            rerenderedOutputs = loss_results[2]

            images = de_normalize(images)

            diffuse, normal, rough, specular = tex2map(brdfs)
            normal = de_normalize((normal.permute(0,2,3,1)+1)/2.0)
            diffuse = de_normalize(gyApplyGamma(diffuse, 1/1.0).permute(0,2,3,1))
            rough = de_normalize(gyApplyGamma(rough, 1/1.0).permute(0,2,3,1))
            specular = de_normalize(gyApplyGamma(specular, 1/2.2).permute(0,2,3,1))

            diffuse_pred, normal_pred, rough_pred, specular_pred = tex2map(brdfs_pred)
            normal_pred = de_normalize((normal_pred.permute(0,2,3,1)+1)/2.0)
            diffuse_pred = de_normalize(gyApplyGamma(diffuse_pred, 1/1.0).permute(0,2,3,1))
            rough_pred = de_normalize(gyApplyGamma(rough_pred, 1/1.0).permute(0,2,3,1))
            specular_pred = de_normalize(gyApplyGamma(specular_pred, 1/2.2).permute(0,2,3,1))

            reproduceImg = de_normalize(reproduceImg)
            rerenderedImg = de_normalize(rerenderedImg.clamp(0,1))
            rerenderedOutputs = de_normalize(rerenderedOutputs.clamp(0,1))

            experiment.log({
                'images': wandb.Image(images[0].type(torch.int64).cpu().permute(1, 2, 0).numpy()),
                'reproduceImg': wandb.Image(reproduceImg[0].type(torch.int64).cpu().permute(1, 2, 0).numpy()),
                'brdf_normal': {
                    'normal_true': wandb.Image(normal[0].type(torch.int64).cpu().numpy()),
                    'normal_pred': wandb.Image(
                        normal_pred[0].type(torch.int64).cpu().numpy()),
                },
                'brdf_diffuse': {
                    'diffuse_true': wandb.Image(diffuse[0].type(torch.int64).cpu().numpy()),
                    'diffuse_pred': wandb.Image(
                        diffuse_pred[0].type(torch.int64).cpu().numpy()),
                },
                'brdf_roughness': {
                    'roughness_true': wandb.Image(
                        rough[0].type(torch.int64).cpu().numpy()),
                    'roughness_pred': wandb.Image(
                        rough_pred[0].type(torch.int64).cpu().numpy()),
                },
                'brdf_specular': {
                    'specular_true': wandb.Image(specular[0].type(torch.int64).cpu().numpy()),
                    'specular_pred': wandb.Image(
                        specular_pred[0].type(torch.int64).cpu().numpy()),
                },
                'rerendered_image': {
                    'rerendered_image_true': wandb.Image(
                        rerenderedImg[0,0:3,...].type(torch.int64).cpu().permute(1, 2, 0).numpy()),
                    'rerendered_image_pred': wandb.Image(
                        rerenderedOutputs[0,0:3,...].type(torch.int64).cpu().permute(1, 2, 0).numpy()),
                },
            })

        torch.cuda.empty_cache()
        count = count +1

    net.train()

    return loss/count
