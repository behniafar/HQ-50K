import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, main, skip=None):
        super().__init__()
        self.main = nn.Sequential(*main)
        self.skip = skip if skip else nn.Identity()

    def forward(self, input):
        return self.main(input) + self.skip(input)


class ResConvBlock(ResidualBlock):
    def __init__(self, c_in, c_mid, c_out):
        skip = None if c_in == c_out else nn.Conv2d(c_in, c_out, 1, bias=False)
        super().__init__([
            nn.Conv2d(c_in, c_mid, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_mid),
            nn.ReLU(),
            nn.Conv2d(c_mid, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(),
        ], skip)


class SkipBlock(nn.Module):
    def __init__(self, main, skip=None):
        super().__init__()
        self.main = nn.Sequential(*main)
        self.skip = skip if skip else nn.Identity()

    def forward(self, input):
        return torch.cat([self.main(input), self.skip(input)], dim=1)

def positional_encoding_2d(x, frequency=10000):
    # image position encoding
    B, C, H, W = x.shape
    x = x.view(B, C, -1).permute(0, 2, 1)  # (B, H*W, C)

    # Create position encodings for rows (H) and columns (W)
    pos_y = torch.arange(0, H, device=x.device).unsqueeze(-1)  # (H, 1)
    pos_x = torch.arange(0, W, device=x.device).unsqueeze(-1)  # (W, 1)

    div_term = torch.arange(0, C//2, device=x.device).unsqueeze(0)  # (1, C//2)

    # Encode y positions (rows)
    pos_encoding_y = pos_y / (frequency ** (2 * div_term / (C//2)))  # (H, C//2)
    pos_embed_y = torch.zeros(H, C//2, device=x.device)
    pos_embed_y[:, 0::2] = torch.sin(pos_encoding_y[:, 0::2])
    pos_embed_y[:, 1::2] = torch.cos(pos_encoding_y[:, 1::2])

        # Encode x positions (columns)
    pos_encoding_x = pos_x / (frequency ** (2 * div_term / (C//2)))  # (W, C//2)
    pos_embed_x = torch.zeros(W, C//2, device=x.device)
    pos_embed_x[:, 0::2] = torch.sin(pos_encoding_x[:, 0::2])
    pos_embed_x[:, 1::2] = torch.cos(pos_encoding_x[:, 1::2])

    # Combine x and y position embeddings
    pos_embed_2d = torch.zeros(H, W, C, device=x.device)
    pos_embed_2d[:, :, :C//2] = pos_embed_y.unsqueeze(1)  # Broadcast y to all columns
    pos_embed_2d[:, :, C//2:] = pos_embed_x.unsqueeze(0)  # Broadcast x to all rows

    # Reshape and add to input
    pos_embed = pos_embed_2d.view(-1, C).unsqueeze(0).expand(B, -1, -1)  # (B, H*W, C)
    x = x + pos_embed
    return x.permute(0, 2, 1).view(B, C, H, W)

class PositionalEncoding2d(nn.Module):
    def __init__(self, frequency = 10000):
        super().__init__()
        self.frequency = frequency

    def forward(self, x):
        return positional_encoding_2d(x, self.frequency)

class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, std=1.):
        super().__init__()
        assert out_features % 2 == 0
        self.weight = nn.Parameter(torch.randn([out_features // 2, in_features]) * std)

    def forward(self, input):
        f = 2 * torch.pi * input @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)

class FourierPositionalEncoding2d(nn.Module):
    def __init__(self, channels, std=1.):
        super().__init__()
        assert channels % 2 == 0
        self.fourier_features = FourierFeatures(2, channels, std)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, -1).permute(0, 2, 1)  # (B, H*W, C)
        grid_x, grid_y = torch.meshgrid(torch.linspace(0, 1, W, device=x.device), torch.linspace(0, 1, H, device=x.device), indexing='ij')
        grid = torch.stack([grid_y, grid_x], dim=-1).view(-1, 2)  # (H*W, 2)
        pos_embed = self.fourier_features(grid)  # (H*W, C)
        pos_embed = pos_embed.unsqueeze(0).expand(B, -1, -1)
        x = x + pos_embed
        return x.permute(0, 2, 1).view(B, C, H, W)

class SelfAttention(nn.Module):
    def __init__(self, channels, heads = 4):
        super().__init__()
        self.attention = nn.MultiheadAttention(channels, heads, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, -1).permute(0, 2, 1)  # (B, H*W, C)
        x = x + self.attention(x, x, x)[0]
        return x.permute(0, 2, 1).view(B, C, H, W)

class FeedForward(nn.Module):
    def __init__(self, channels, expansion=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels * expansion, 1),
            nn.GELU(),
            nn.Conv2d(channels * expansion, channels, 1),
        )

    def forward(self, x):
        return self.net(x)

class StaticMoEBlock(nn.Module):
    def __init__(self, channels, num_experts=8, k = 2, expansion=4):
        super().__init__()
        self.k = k
        self.experts = nn.ModuleList([
            FeedForward(channels, expansion) for _ in range(num_experts)
        ])
        self.router = nn.Conv2d(channels, num_experts, 1)

    def forward(self, x):
        # NOTE: maximize the gate scores std during training
        # NOTE: Normal DAMoE do not use the router output as expert output's factor, but I did so to keep the gradient to flow properly.
        B, C, H, W = x.shape
        self.router_scores = F.softmax(self.router(x), dim=1)  # (B, num_experts, H, W)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)  # (B, num_experts, C, H, W)
        topk_scores, topk_indices = torch.topk(self.router_scores, k=self.k, dim=1)  # (B, k, H, W)
        mask = torch.zeros_like(self.router_scores)
        mask.scatter_(1, topk_indices, self.router_scores)
        output = (expert_outputs * mask.unsqueeze(2)).sum(dim=1) # (B, C, H, W)
        return output

class DynamicMoEBlock(nn.Module):
    def __init__(self, channels, num_experts=4, normal_active_experts = 1, expansion=4):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            FeedForward(channels, expansion) for _ in range(num_experts)
        ])
        self.normal_active_experts = normal_active_experts
        self.router = nn.Conv2d(channels, num_experts, 1)
        self.min_score = 0.5

    def forward(self, x):
        # NOTE: maximize the gate scores std during training
        # NOTE: Normal DAMoE do not use the router output as expert output's factor, but I did so to keep the gradient to flow properly.
        # TODO: adjust min_score according to normal_active_experts
        B, C, H, W = x.shape
        self.router_scores = F.sigmoid(self.router(x), dim=1)  # (B, num_experts, H, W)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)  # (B, num_experts, C, H, W)
        mask = torch.zeros_like(self.router_scores)
        selected_experts = (self.router_scores > self.min_score).float().sum(dim=1, keepdim=True)  # (B, 1, H, W)
        mask = torch.zeros_like(self.router_scores)
        mask[self.router_scores > self.min_score] = self.router_scores[self.router_scores > self.min_score]
        top_one_scores, top_one_indices = torch.topk(self.router_scores, k=1, dim=1)  # (B, 1, H, W)
        top_one = torch.zeros_like(self.router_scores)
        top_one.scatter_(1, top_one_indices, top_one_scores)
        mask = torch.where(selected_experts < 1, top_one, mask)
        output = (expert_outputs * mask.unsqueeze(2)).sum(dim=1) # (B, C, H, W)
        return output

MoEBlock = StaticMoEBlock  # default MoEBlock
    
# TODO: optimize MoEBlock, i mean the model must use selected experts only for each pixel that selects them

class StaticDAMoEBlock(nn.Module):
    def __init__(self, channels, num_experts=8, k = 2, expansion=4, heads=4):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(channels)
        self.attention = SelfAttention(channels, heads)
        self.norm2 = nn.BatchNorm2d(channels)
        self.moe = StaticMoEBlock(channels, num_experts, k, expansion)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.moe(self.norm2(x))
        return x

class DynamicDAMoEBlock(nn.Module):
    def __init__(self, channels, num_experts=4, normal_active_experts = 1, expansion=4, heads=4):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(channels)
        self.attention = SelfAttention(channels, heads)
        self.norm2 = nn.BatchNorm2d(channels)
        self.moe = DynamicMoEBlock(channels, num_experts, normal_active_experts, expansion)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.moe(self.norm2(x))
        return x

DAMoEBlock = StaticDAMoEBlock  # default DAMoEBlock
