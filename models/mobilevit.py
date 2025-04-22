import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class MV2Block(nn.Module):
    def __init__(self, inp, out, stride=1, expansion=4):
        super().__init__()
        self.stride = stride
        hidden_dim = inp * expansion
        
        # Use self.net for main branch
        self.net = nn.Sequential(
            # Pointwise
            nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(),
            # Depthwise
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(),
            # Pointwise-linear
            nn.Conv2d(hidden_dim, out, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out)
        )
        
        # Use shortcut if inp and out dimensions are the same
        self.use_shortcut = (stride == 1 and inp == out)

    def forward(self, x):
        if self.use_shortcut:
            return x + self.net(x)
        else:
            return self.net(x)


class MobileViTAttention(nn.Module):
    def __init__(self, dim, depth, heads=4, dim_head=64, mlp_dim=128):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(dim),
                nn.MultiheadAttention(embed_dim=dim, num_heads=heads),
                nn.LayerNorm(dim),
                nn.Sequential(
                    nn.Linear(dim, mlp_dim),
                    nn.SiLU(),
                    nn.Linear(mlp_dim, dim)
                )
            ]))

    def forward(self, x):
        for norm1, attn, norm2, mlp in self.layers:
            x_norm = norm1(x)
            attn_out, _ = attn(x_norm, x_norm, x_norm)
            x = x + attn_out
            x = x + mlp(norm2(x))
        return x


class MobileViTBlock(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, heads=4, dim_head=64):
        super().__init__()
        self.ph, self.pw = patch_size
        
        self.conv1 = nn.Conv2d(channel, channel, kernel_size=kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv2d(channel, dim, kernel_size=1)
        
        self.transformer = MobileViTAttention(dim=dim, depth=depth, heads=heads, dim_head=dim_head, mlp_dim=mlp_dim)
        
        self.conv3 = nn.Conv2d(dim, channel, kernel_size=1)
        self.conv4 = nn.Conv2d(2 * channel, channel, kernel_size=kernel_size, padding=kernel_size//2)
    
    def forward(self, x):
        # Local representation
        y = self.conv1(x)
        y = self.conv2(y)
        
        # Global representations
        _, _, h, w = y.shape
        y = rearrange(y, 'b c (h ph) (w pw) -> b (h w) (ph pw c)', ph=self.ph, pw=self.pw)
        y = self.transformer(y)
        y = rearrange(y, 'b (h w) (ph pw c) -> b c (h ph) (w pw)', h=h//self.ph, w=w//self.pw, ph=self.ph, pw=self.pw)
        
        # Fusion
        y = self.conv3(y)
        y = torch.cat([x, y], dim=1)
        y = self.conv4(y)
        return y


class MobileViT(nn.Module):
    def __init__(self, image_size=(240, 320), dims=[96, 120, 144], channels=[16, 32, 48, 64, 80, 96, 384], 
                 num_classes=1, expansion=4, kernel_size=3, patch_size=(2, 2), depths=[2, 4, 3]):
        super().__init__()
        ih, iw = image_size
        ph, pw = patch_size
        
        # Check if image dimensions are divisible by patch size
        assert ih % ph == 0 and iw % pw == 0
        
        # Initial conv layer
        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=3, stride=2, padding=1)
        
        # MV2 blocks
        self.mv2 = nn.ModuleList([])
        self.mv2.append(MV2Block(channels[0], channels[1], 1, expansion))
        self.mv2.append(MV2Block(channels[1], channels[2], 2, expansion))
        self.mv2.append(MV2Block(channels[2], channels[3], 1, expansion))
        self.mv2.append(MV2Block(channels[3], channels[4], 2, expansion))
        self.mv2.append(MV2Block(channels[4], channels[5], 1, expansion))
        
        # MobileViT blocks
        self.mvit = nn.ModuleList([])
        self.mvit.append(MobileViTBlock(dims[0], depths[0], channels[5], kernel_size, patch_size, int(dims[0]*2)))
        self.mvit.append(MobileViTBlock(dims[1], depths[1], channels[5], kernel_size, patch_size, int(dims[1]*4)))
        self.mvit.append(MobileViTBlock(dims[2], depths[2], channels[6], kernel_size, patch_size, int(dims[2]*4)))
        
        # Final layers
        self.conv2 = nn.Conv2d(channels[6], channels[6], kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels[6], num_classes)
        
    def forward(self, x):
        # Initial conv
        x = self.conv1(x)
        
        # MV2 blocks
        x = self.mv2[0](x)
        x = self.mv2[1](x)
        x = self.mv2[2](x)
        x = self.mv2[3](x)
        x = self.mv2[4](x)
        
        # MobileViT blocks
        x = self.mvit[0](x)
        x = self.mvit[1](x)
        x = self.mvit[2](x)
        
        # Final layers
        x = self.conv2(x)
        x = self.pool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        
        return x


class TemporalFusionTransformer(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads, dropout=0.1, num_layers=2):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Input projection
        self.input_projection = nn.Linear(input_size, hidden_size)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_projection = nn.Linear(hidden_size, 1)  # Output is steering angle
        
    def forward(self, x, mask=None):
        # x shape: [batch_size, seq_len, input_size]
        
        # Project input to hidden size
        x = self.input_projection(x)
        
        # Transpose for transformer: [seq_len, batch_size, hidden_size]
        x = x.transpose(0, 1)
        
        # Apply transformer encoder
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        # Transpose back: [batch_size, seq_len, hidden_size]
        x = x.transpose(0, 1)
        
        # Get the last time step
        x = x[:, -1, :]
        
        # Project to output
        x = self.output_projection(x)
        
        return x.squeeze(-1)  # Return [batch_size]


class VisionTemporalModel(nn.Module):
    def __init__(self, seq_len=10, vision_features=384, state_features=3):
        super().__init__()
        
        # Vision model (MobileViT)
        self.vision_model = MobileViT(num_classes=vision_features)
        
        # Combined features
        combined_features = vision_features + state_features
        
        # Temporal model (TFT)
        self.temporal_model = TemporalFusionTransformer(
            input_size=combined_features,
            hidden_size=256,
            num_heads=4,
            dropout=0.1,
            num_layers=2
        )
        
    def forward(self, images, states, mask=None):
        # images shape: [batch_size, seq_len, 3, height, width]
        # states shape: [batch_size, seq_len, state_features]
        
        batch_size, seq_len = images.shape[0], images.shape[1]
        
        # Process each image in the sequence
        vision_features = []
        for t in range(seq_len):
            # Get current timestep images
            current_images = images[:, t]
            
            # Extract vision features
            features = self.vision_model(current_images)
            vision_features.append(features)
        
        # Stack vision features
        vision_features = torch.stack(vision_features, dim=1)  # [batch_size, seq_len, vision_features]
        
        # Combine vision features with state features
        combined_features = torch.cat([vision_features, states], dim=2)
        
        # Process with temporal model
        output = self.temporal_model(combined_features, mask)
        
        return output  # Steering angle prediction
