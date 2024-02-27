import argparse
import logging
import os
import time

import numpy as np
import torch
from PIL import Image
import imageio
from tqdm import tqdm
from pathlib import Path

from utils.data_loading_brdf import BasicDataset
from utils.utils import de_normalize, preprocess,log_transform
from torch.utils.data import DataLoader
from utils.rendererMG import tex2map
from utils.rendererMG_util import gyApplyGamma


from unet import UNet
from rim import RIM, ConvRNN, model_wrapped

dir_img_predict = Path('./dataset/')
checkpoint_filename = Path('./ckpt/pretrained_model.pth')

def predict(net, dataloader, args, device):
    net.eval()
    num_pred_batches = len(dataloader)
    count = 0

    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_pred_batches, desc='Predicting...', unit='batch', leave=False):
        images = batch['image']
        imgname = batch['image_name']
        images = images.to(device=device, dtype=torch.float32)

        if args.correctGamma:
            images = torch.pow(images, 2.2)
            print('\t correctGamma: ', args.correctGamma)

        if args.useLog:
            delta = torch.tensor(0.01).to(device=device, dtype=torch.float32)
            images_log = log_transform(images, delta)
            print('\t useLog: ', args.useLog)
        else:
            images_log = images

        eta = preprocess(images_log)

        with torch.no_grad():
            # predict the brdfs
            start = time.time()
            brdfs_preds = net(eta, images, args, imgname[0])
            end = time.time()
            print('time', end - start)
            brdfs_pred = brdfs_preds[1]

            diffuse_pred, normal_pred, rough_pred, specular_pred = tex2map(brdfs_pred)
            normal_pred = de_normalize((normal_pred.permute(0, 2, 3, 1) + 1) / 2.0)[0]
            diffuse_pred = de_normalize(gyApplyGamma(diffuse_pred, 1 /2.2).permute(0, 2, 3, 1))[0]
            rough_pred = de_normalize(gyApplyGamma(rough_pred, 1 /1.0).permute(0, 2, 3, 1))[0]
            specular_pred = de_normalize(gyApplyGamma(specular_pred, 1 /2.2).permute(0, 2, 3, 1))[0]

            maps = torch.cat((diffuse_pred, specular_pred, rough_pred, normal_pred), axis=1)
            maps = maps.cpu().numpy()

            out_folder_path = './results'
            if os.path.isdir(out_folder_path) is False:
                os.mkdir(out_folder_path)
            out_img_path = out_folder_path + '/'
            if os.path.isdir(out_img_path) is False:
                os.mkdir(out_img_path)

            imageio.imwrite(out_img_path + imgname[0] + '_map.png', maps.astype(np.uint8))

        torch.cuda.empty_cache()
        count = count + 1

    return


def get_args():
    parser = argparse.ArgumentParser(description='Predict brdfs from photos')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--scale', '-s', type=float, default=1.0,
                        help='Scale factor for the input images')

    parser.add_argument("--useLog", dest="useLog", action="store_true", help="Use the log for input")
    parser.set_defaults(useLog=False)

    parser.add_argument('--nbDiffuseRendering', '--nrDiff', type=int, default=3,
                        help="Number of diffuse renderings in the rendering loss")
    parser.add_argument('--nbSpecularRendering', '--nrSpec', type=int, default=6,
                        help="Number of specular renderings in the rendering loss")
    parser.add_argument("--includeDiffuse", dest="includeDiffuse", action="store_true",
                        help="Include the diffuse term in the specular renderings of the rendering loss ?")
    parser.set_defaults(includeDiffuse=True)
    parser.add_argument("--correctGamma", dest="correctGamma", action="store_true", help="correctGamma ? ?")
    parser.set_defaults(correctGamma=False)

    parser.add_argument('--recurrent_layer', type=str, choices=['gru', 'indrnn'], default='gru',
                        help='Type of recurrent input')
    parser.add_argument('--n_steps', type=int, default=6, help='Number of RIM steps')
    parser.add_argument('--loss', '-ls', type=str, default="rendering_loss_l1", help='specify the loss function from one of these: l1, l2, rendering_loss_l1, rendering_loss_l2')
    return parser.parse_args()


def get_output_filenames(args):
    def _generate_name(fn):
        split = os.path.splitext(fn)
        return f'{split[0]}_OUT{split[1]}'

    return args.output or list(map(_generate_name, args.input))


def mask_to_image(mask: np.ndarray):
    if mask.ndim == 2:
        return Image.fromarray((mask * 255).astype(np.uint8))
    elif mask.ndim == 3:
        return Image.fromarray((np.argmax(mask, axis=0) * 255 / mask.shape[0]).astype(np.uint8))


if __name__ == '__main__':
    args = get_args()

    im_channels = (9+1)*2
    conv_nd = 2
    rnn = ConvRNN(input_size=im_channels, recurrent_layer=args.recurrent_layer, conv_dim=conv_nd)
    model = RIM(rnn, n_steps=args.n_steps)
    nr_input_channels = 3
    nr_output_channels = 9+1
    model = model_wrapped(preprocess_net=UNet(nr_input_channels, nr_output_channels), main_net=model)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {checkpoint_filename}')
    logging.info(f'Using device {device}')

    model.to(device=device)
    model.load_state_dict(torch.load(checkpoint_filename, map_location=device))

    logging.info('Model loaded!')
    dataset_predict = BasicDataset(dir_img_predict, scale=1, train=False)
    n_predict = len(dataset_predict)

    batch_size = 1
    loader_args = dict(batch_size=batch_size, num_workers=0, pin_memory=True)
    predict_loader = DataLoader(dataset_predict, shuffle=False, **loader_args)

    predict(model, predict_loader, args, device)