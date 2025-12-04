import os
import torch
from torch import optim
from tqdm import trange, tqdm
import torch.nn.functional as F
from torchvision import utils as U
from torchvision.transforms import functional as TF

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, dataloader, optimizer, repeat_factor = 1, ema = .9):
    """
    Trains the model for one epoch.
    Note: if it's too slow to load data, repeat them to increase batch size.
    
    Args:
        model: The neural network model to be trained.
        dataloader: DataLoader providing the training data.
        optimizer: Optimizer for updating model parameters.
        repeat_factor: Factor to repeat the data for increasing batch size.
        ema: Exponential moving average factor for loss smoothing.
    """
    model.train()
    total_loss = None
    bar = tqdm(dataloader)
    for reals, _ in bar:
        optimizer.zero_grad()
        reals = reals.to(device)

        loss = model.loss(reals, repeat_factor=repeat_factor)
        total_loss = loss.item() * (1 - ema) + total_loss * ema if total_loss is not None else loss.item()
        bar.set_description(f"Loss: {total_loss:.6f}")

        loss.backward()
        optimizer.step()

@torch.no_grad()
def demo(model, size, steps, n = 3, filename = None):
    torch.manual_seed(0)

    noise = torch.randn([n**2, 3, *size], device=device)
    fakes = model.sample(noise, steps=steps)

    grid = U.make_grid(fakes, n).cpu()
    if filename is None:
        filename = f'demo.png'
    grid = TF.to_pil_image(grid.clamp(0, 1))
    grid.save(filename)
    tqdm.write(f'Demo saved to {os.path.abspath(filename)}\n')
    return grid

class RandomSquareMask:
    def __init__(self, min_size=16, max_size=64):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, shape, device=device):
        shape = list(shape)
        shape[1] = 1
        B, C, H, W = shape
        masks = torch.zeros(shape, device=device)
        for i in range(B):
            size = torch.randint(self.min_size, self.max_size + 1, (1,)).item()
            top = torch.randint(0, H - size + 1, (1,)).item()
            left = torch.randint(0, W - size + 1, (1,)).item()
            masks[i, :, top:top + size, left:left + size] = 1
        return masks

class RandomCircleMask:
    def __init__(self, min_radius=8, max_radius=32):
        self.min_radius = min_radius
        self.max_radius = max_radius

    def __call__(self, shape, device=device):
        shape = list(shape)
        shape[1] = 1
        B, C, H, W = shape
        masks = torch.zeros(shape, device=device)
        Y, X = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
        for i in range(B):
            radius = torch.randint(self.min_radius, self.max_radius + 1, (1,)).item()
            center_y = torch.randint(radius, H - radius + 1, (1,)).item()
            center_x = torch.randint(radius, W - radius + 1, (1,)).item()
            dist_sq = (X - center_x) ** 2 + (Y - center_y) ** 2
            masks[i, :, dist_sq <= radius ** 2] = 1
        return masks

def masked_train(model, dataloader, optimizer, mask_maker = [RandomSquareMask(), RandomCircleMask()], repeat_factor = 1, ema = .9):
    """
    Trains the model for one epoch with masked images.
    Note: if it's too slow to load data, repeat them to increase batch size.
    
    Args:
        model: The neural network model to be trained.
        dataloader: DataLoader providing the training data.
        optimizer: Optimizer for updating model parameters.
        mask_maker: List of mask generators to create random masks.
        repeat_factor: Factor to repeat the data for increasing batch size.
        ema: Exponential moving average factor for loss smoothing.
    """
    model.train()
    total_loss = None
    bar = tqdm(dataloader)
    for reals, _ in bar:
        optimizer.zero_grad()
        reals = reals.to(device)

        # Generate random masks
        masks = torch.zeros_like(reals)
        for mask_gen in mask_maker:
            masks = torch.max(masks, mask_gen(reals.shape, device=device))
        
        loss = model.loss(reals, masks=masks, repeat_factor=repeat_factor)
        loss.backward()
        optimizer.step()
        
        total_loss = loss.item() * (1 - ema) + total_loss * ema if total_loss is not None else loss.item()
        bar.set_description(f"Loss: {total_loss:.6f}")

@torch.no_grad()
def masked_demo(model, images, masks = None, steps=50, filename=None):
    torch.manual_seed(0)

    images = images.to(device)
    if masks is None:
        masks = torch.ones_like(images)
    else:
        masks = masks.to(device)

    noise = torch.randn_like(images)
    fakes = model.sample(images * (1 - masks) + noise * masks, mask=masks, steps=steps)

    B = images.shape[0]
    grid = U.make_grid(torch.cat([images, fakes], dim=0), int(B**0.5)).cpu()
    if filename is None:
        filename = f'masked_demo.png'
    grid = TF.to_pil_image(grid.clamp(0, 1))
    grid.save(filename)
    tqdm.write(f'Masked demo saved to {os.path.abspath(filename)}\n')
    return grid

def save_model(model, optimizer, epoch, filename='model.pth'):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }, filename)