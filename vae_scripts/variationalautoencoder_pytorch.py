import os
import torch
import numpy as np
import torch.nn as nn
from torch.optim import Adam
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from mpl_toolkits.axes_grid1 import ImageGrid
from torchvision.utils import save_image, make_grid
import logging
from torch.utils.tensorboard import SummaryWriter
import argparse
import json
from tqdm import tqdm

from research.utils import set_logger

parser = argparse.ArgumentParser(description='Training VAE for MNIST')
parser.add_argument('--checkpoint_dir', default='/efs/user_folders/gilad/vae_mnist/debug4', type=str, help='checkpoint dir')
# training
parser.add_argument('--epochs', default=50, type=int, help='Number of training epochs')
# loss
parser.add_argument('--alpha', default=1.0, type=float, help='contribution of KLD loss')

parser.add_argument('--mode', default='null', type=str, help='to bypass pycharm bug')
parser.add_argument('--port', default='null', type=str, help='to bypass pycharm bug')
parser.add_argument('--host', default='null', type=str, help='to bypass pycharm bug')

args = parser.parse_args()
torch.autograd.set_detect_anomaly(True)
x_dim = 784

# dumping args to txt file
os.makedirs(args.checkpoint_dir, exist_ok=True)
with open(os.path.join(args.checkpoint_dir, 'commandline_args.txt'), 'w') as f:
    json.dump(args.__dict__, f, indent=2)
log_file = os.path.join(args.checkpoint_dir, 'log.log')
set_logger(log_file)
logger = logging.getLogger()

# create a transofrm to apply to each datapoint
transform = transforms.Compose([transforms.ToTensor()])

# download the MNIST datasets
path = '/efs/datasets'
train_dataset = MNIST(path, transform=transform, download=True)
test_dataset = MNIST(path, transform=transform, download=True)

# create train and test dataloaders
batch_size = 100
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

train_writer = SummaryWriter(os.path.join(args.checkpoint_dir, 'train'))
val_writer   = SummaryWriter(os.path.join(args.checkpoint_dir, 'val'))
test_writer  = SummaryWriter(os.path.join(args.checkpoint_dir, 'test'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# get 25 sample training images for visualization
images, labels = list(train_loader)[0]

num_samples = 25
sample_images = [images[i, 0] for i in range(num_samples)]

fig = plt.figure(figsize=(5, 5))
grid = ImageGrid(fig, 111, nrows_ncols=(5, 5), axes_pad=0.1)

for ax, im in zip(grid, sample_images):
    ax.imshow(im, cmap='gray')
    ax.axis('off')

plt.show()


class VAE(nn.Module):

    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=200, device=device):
        super(VAE, self).__init__()

        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, latent_dim),
            nn.LeakyReLU(0.2)
        )

        # latent mean and variance
        self.mean_layer = nn.Linear(latent_dim, 2)
        self.logvar_layer = nn.Linear(latent_dim, 2)

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(2, latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x)
        mean, logvar = self.mean_layer(x), self.logvar_layer(x)
        return mean, logvar

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(device)
        z = mean + var * epsilon
        return z

    def decode(self, x):
        return self.decoder(x)

    # def forward(self, x):
    #     mean, logvar = self.encode(x)
    #     z = self.reparameterization(mean, logvar)
    #     x_hat = self.decode(z)
    #     return x_hat, mean, log_var

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))
        x_hat = self.decode(z)
        return x_hat, mean, log_var


model = VAE().to(device)
optimizer = Adam(model.parameters(), lr=1e-3)


def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    KLD = - 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    loss = reproduction_loss + args.alpha * KLD
    return {'loss': loss, 'reproduction_loss': reproduction_loss, 'KLD': KLD, 'alpha_KLD': args.alpha * KLD}


def train():
    model.train()
    overall_losses = {'loss': 0.0, 'reproduction_loss': 0.0, 'KLD': 0.0, 'alpha_KLD': 0.0}
    vals = {'mean1': 0.0, 'mean2': 0.0, 'log_var1': 0.0, 'log_var2': 0.0, 'var1': 0.0, 'var2': 0.0, 'abs_var': 0.0}
    cnt = 0
    for batch_idx, (x, _) in enumerate(train_loader):
        x = x.view(batch_size, x_dim).to(device)

        optimizer.zero_grad()

        x_hat, mean, log_var = model(x)
        var = torch.exp(0.5 * log_var)
        losses = loss_function(x, x_hat, mean, log_var)
        for key in overall_losses.keys():
            overall_losses[key] += losses[key].item()
        vals['mean1'] += mean[:, 0].sum()
        vals['mean2'] += mean[:, 1].sum()
        vals['log_var1'] += log_var[:, 0].sum()
        vals['log_var2'] += log_var[:, 1].sum()
        vals['var1'] += var[:, 0].sum()
        vals['var2'] += var[:, 1].sum()
        vals['abs_var'] += torch.linalg.vector_norm(var, dim=1).sum()
        cnt += len(x)

        losses['loss'].backward()
        optimizer.step()

    vals.update(overall_losses)
    for key in vals.keys():
        val = vals[key] / cnt
        train_writer.add_scalar(key, val, epoch + 1)
        logger.info('Epoch #{} (TRAIN): {}\t{:.2f}'.format(epoch + 1, key, val))

    return overall_losses


def validate():
    model.eval()
    overall_losses = {'loss': 0.0, 'reproduction_loss': 0.0, 'KLD': 0.0, 'alpha_KLD': 0.0}
    vals = {'mean1': 0.0, 'mean2': 0.0, 'log_var1': 0.0, 'log_var2': 0.0, 'var1': 0.0, 'var2': 0.0, 'abs_var': 0.0}
    cnt = 0
    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(test_loader):
            x = x.view(batch_size, x_dim).to(device)

            x_hat, mean, log_var = model(x)
            var = torch.exp(0.5 * log_var)
            losses = loss_function(x, x_hat, mean, log_var)
            for key in overall_losses.keys():
                overall_losses[key] += losses[key].item()
            vals['mean1'] += mean[:, 0].sum()
            vals['mean2'] += mean[:, 1].sum()
            vals['log_var1'] += log_var[:, 0].sum()
            vals['log_var2'] += log_var[:, 1].sum()
            vals['var1'] += var[:, 0].sum()
            vals['var2'] += var[:, 1].sum()
            vals['abs_var'] += torch.linalg.vector_norm(var, dim=1).sum()
            cnt += len(x)

    vals.update(overall_losses)
    for key in vals.keys():
        val = vals[key] / cnt
        val_writer.add_scalar(key, val, epoch + 1)
        logger.info('Epoch #{} (VAL): {}\t{:.2f}'.format(epoch + 1, key, val))

    return vals


def save_current_state():
    torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, 'ckpt_epoch_{}.pth'.format(epoch)))


epoch = 0
global_step = 0
logger.info('Eval epoch #{}'.format(epoch + 1))
validate()
logger.info('Start training from epoch #{} for {} epochs'.format(epoch + 1, args.epochs))
for epoch in tqdm(range(epoch, epoch + args.epochs), total=args.epochs):
    train()
    validate()
save_current_state()


def generate_digit(mean, var):
    z_sample = torch.tensor([[mean, var]], dtype=torch.float).to(device)
    x_decoded = model.decode(z_sample)
    digit = x_decoded.detach().cpu().reshape(28, 28) # reshape vector to 2d array
    plt.title(f'[{mean},{var}]')
    plt.imshow(digit, cmap='gray')
    plt.axis('off')
    plt.savefig(os.path.join(args.checkpoint_dir, f'fig_for_{mean}_{var}.png'), dpi=300)
    plt.show()


#img1: mean0, var1 / img2: mean1, var0
generate_digit(0.0, 1.0), generate_digit(1.0, 0.0)


def plot_latent_space(model, scale=5.0, n=25, digit_size=28, figsize=15):
    # display a n*n 2D manifold of digits
    figure = np.zeros((digit_size * n, digit_size * n))

    # construct a grid
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = torch.tensor([[xi, yi]], dtype=torch.float).to(device)
            x_decoded = model.decode(z_sample)
            digit = x_decoded[0].detach().cpu().reshape(digit_size, digit_size)
            figure[i * digit_size : (i + 1) * digit_size, j * digit_size : (j + 1) * digit_size,] = digit

    plt.figure(figsize=(figsize, figsize))
    plt.title('VAE Latent Space Visualization')
    start_range = digit_size // 2
    end_range = n * digit_size + start_range
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("mean, z [0]")
    plt.ylabel("var, z [1]")
    plt.imshow(figure, cmap="Greys_r")
    plt.savefig(os.path.join(args.checkpoint_dir, f'fig_for_5x5_scale.png'), dpi=300)
    plt.show()
