import argparse
import logging
import sys

from pathlib import Path
import torch
import wandb
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.data_loading_brdf import BasicDataset
from evaluate import evaluate
from unet import UNet
from rim import RIM, ConvRNN, ConvRNN_highlightaware, model_wrapped
from utils.loss import loss_func
from utils.utils import preprocess, log_transform


dir_img_tr = Path('./training_dataset/')
dir_img_va = Path('./testing_dataset/')
dir_checkpoint = Path('./checkpoints/')

class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0-self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_net(net,
              device,
              epochs: int = 5,
              batch_size: int = 1,
              learning_rate: float = 0.001,
              val_percent: float = 0.1,
              save_checkpoint: bool = True,
              img_scale: float = 1,
              amp: bool = False):

    dataset_train = BasicDataset(dir_img_tr, img_scale)
    dataset_val = BasicDataset(dir_img_va, img_scale)
    n_train = len(dataset_train)
    n_val = len(dataset_val)
    

    loader_args = dict(batch_size=batch_size, num_workers=8, pin_memory=True)
    train_loader = DataLoader(dataset_train, shuffle=True, **loader_args)
    val_loader = DataLoader(dataset_val, shuffle=False, drop_last=True, **loader_args)

    experiment = wandb.init(project='BRDFNet', resume='allow', anonymous='must', dir=str(dir_img_tr.parent.parent))
    experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                                  val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale,
                                  amp=amp))

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    optimizer = optim.Adam(net.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=0)
    global_step = 0

    
    # Begin training
    for epoch in range(epochs):
        count=0
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='itr') as pbar:
            for batch in train_loader:
                count = count+1
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
                optimizer.zero_grad()
                net.zero_grad()
                brdfs_cat_both = net(eta, images, args)
                brdfs_cat = brdfs_cat_both[1]
                brdfs_cat_unet = brdfs_cat_both[0]

                loss = 0
                loss_results = loss_func(brdfs, brdfs_cat, images, args)
                loss = loss_func(brdfs, brdfs_cat_unet, images, args)[3] + loss_results[3]
                loss.backward()
                optimizer.step()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'train loss_laststep': loss_results[3].item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                division_step = (n_train // (args.val * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in net.named_parameters():
                            tag = tag.replace('/', '.')
                            histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())

                        val_score = evaluate(net, val_loader, args, device, experiment)
                        logging.info('Validation loss: {}'.format(val_score))
                        experiment.log({
                            'learning rate': optimizer.param_groups[0]['lr'],
                            'validation loss': val_score,
                            'step': global_step,
                            'epoch': epoch,
                            **histograms
                        })
                        if save_checkpoint:
                            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
                            torch.save(net.state_dict(), str(dir_checkpoint / 'checkpoint_BRDFNet_{epoch}_{step}.pth'.format(epoch=epoch+1, step=count)))
                            logging.info(f'Checkpoint {epoch + 1} at step {count} saved!')
        experiment.log({
            'epoch_loss': epoch_loss/count,
        })


def get_args():
    parser = argparse.ArgumentParser(description='BRDFs acquisition by using a neural network')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=0.00002,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=1.0, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--loss', '-ls', type=str, default="rendering_loss_l1", help='specify the loss function from one of these: l1, l2, rendering_loss_l1, rendering_loss_l2')
    parser.add_argument("--useLog", dest="useLog", action="store_true", help="Use the log for input")
    parser.set_defaults(useLog=False)
    
    parser.add_argument('--nbDiffuseRendering', '--nrDiff', type=int, default=3, help="Number of diffuse renderings in the rendering loss")
    parser.add_argument('--nbSpecularRendering', '--nrSpec', type=int, default=6, help="Number of specular renderings in the rendering loss")
    parser.add_argument("--includeDiffuse", dest="includeDiffuse", action="store_true", help="Include the diffuse term in the specular renderings of the rendering loss ?")
    parser.set_defaults(includeDiffuse=True)
    parser.add_argument("--correctGamma", dest="correctGamma", action="store_true", help="correctGamma ? ?")
    parser.set_defaults(correctGamma=False)
    parser.add_argument('--recurrent_layer', type=str, choices=['gru', 'indrnn'], default='gru',
                        help='Type of recurrent input')
    parser.add_argument('--n_steps', type=int, default=6, help='Number of RIM steps')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    im_channels = 10 * 2
    conv_nd = 2
    rnn = ConvRNN(input_size=im_channels, recurrent_layer=args.recurrent_layer, conv_dim=conv_nd)
    model = RIM(rnn, n_steps=args.n_steps)
    nr_input_channels = 3
    nr_output_channels = 10
    model = model_wrapped(preprocess_net=UNet(nr_input_channels, nr_output_channels), main_net=model)

    logging.info(f'Network:\n'
                 f'\t{rnn.input_size} input channels\n'
                 f'\t{rnn.recurrent_layer} recurrent_layer\n')
    print('Number of trainable paras: ', count_parameters(model))

    if args.load:
        print('load model from checkpoint')
        model.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)
    try:
        train_net(net=model,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  learning_rate=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100,
                  amp=args.amp)
    except KeyboardInterrupt:
        torch.save(model.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        sys.exit(0)
