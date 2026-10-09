import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class DPTConnector(nn.Module):
    def __init__(self, output_dim, feature_paths, feature_channels, fusion_hidden_size=256, name=""):
        super().__init__()
        self.name = name
        self.output_dim = output_dim
        self.feature_paths = feature_paths
        self.feature_channels = feature_channels
        
        # Reassemble blocks - project each feature to common dimension
        self.reassemble_blocks = nn.ModuleList([
            nn.Conv2d(in_ch, fusion_hidden_size, kernel_size=3, padding=1, bias=False)
            for in_ch in self.feature_channels
        ])
        
        # Fusion blocks - progressive upsampling
        self.fusion3 = FeatureFusionBlock(fusion_hidden_size, combine=False)
        self.fusion2 = FeatureFusionBlock(fusion_hidden_size)
        self.fusion1 = FeatureFusionBlock(fusion_hidden_size)
        self.fusion0 = FeatureFusionBlock(fusion_hidden_size)
        
        # Final prediction head
        self.head = nn.Sequential(
            nn.Conv2d(fusion_hidden_size, fusion_hidden_size // 2, 3, padding=1),
            nn.GroupNorm(16, fusion_hidden_size // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(fusion_hidden_size // 2, output_dim, 1)
        )
        self.initialize_weights()
    
    def forward(self, intermediate_features):
        """
        Args:
            intermediate_features: dict mapping layer paths to features
            Features expected in order: [coarsest, ..., finest]
        
        Returns:
            tensor of shape (B, output_dim, H, W)
        """
        # Extract features (coarsest to finest)
        features = [intermediate_features[path] for path in self.feature_paths]
        
        # Reassemble - project to common dimension
        layer0 = self.reassemble_blocks[0](features[0])  # Coarsest
        layer1 = self.reassemble_blocks[1](features[1])
        layer2 = self.reassemble_blocks[2](features[2])
        layer3 = self.reassemble_blocks[3](features[3])  # Finest

        # Progressive fusion (bottom-up: coarse to fine)
        x = self.fusion3(layer0)              # Refine coarsest, then upsample
        x = self.fusion2(x, layer1)           # Fuse with layer1, then upsample
        x = self.fusion1(x, layer2)           # Fuse with layer2, then upsample
        x = self.fusion0(x, layer3)           # Fuse with finest
        
        # Final prediction head
        out = self.head(x)
        
        return out
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        save_path = os.path.join(save_directory, f"{self.name}.bin")
        torch.save(self.state_dict(), save_path)
        return f"Model saved to {save_path}"
    
    def from_pretrained(self, load_directory):
        load_path = os.path.join(load_directory, f"{self.name}.bin")
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"No model found at {load_path}")
        weight_dev = next(self.parameters()).device
        self.load_state_dict(torch.load(load_path, map_location=weight_dev))
        return f"Model loaded from {load_path}"


class FeatureFusionBlock(nn.Module):
    """Feature fusion block from DPT."""
    
    def __init__(self, features, combine=True):
        super().__init__()
        if combine:
            self.resConfUnit1 = ResidualConvUnit(features)
        self.resConfUnit2 = ResidualConvUnit(features)
        
    def forward(self, *xs):
        """ Forward pass. """
        output = xs[0]
        
        # If skip connection provided, add it after refinement
        if len(xs) == 2:
            if output.shape != xs[1].shape:
                output = F.interpolate(
                    output, size=xs[1].shape[2:], mode="bilinear", align_corners=True
                )
            output = output + self.resConfUnit1(xs[1])
        
        # Refine
        output = self.resConfUnit2(output)
        
        return output


class ResidualConvUnit(nn.Module):
    """Residual convolution module from DPT."""
    
    def __init__(self, features):
        super().__init__()
        
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.gn1 = nn.GroupNorm(32, features)
        self.gn2 = nn.GroupNorm(32, features)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        out = self.relu(x)
        out = self.conv1(out)
        out = self.gn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.gn2(out)
        
        return out + x