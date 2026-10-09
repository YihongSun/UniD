import os
import torch
import torch.nn as nn

class TaskProjector(nn.Module):
    def __init__(self, input_conv, output_dim, hidden_dims=None, name=""):
        """ Per-pixel MLP that projects from R^M to R^N
        """
        super(TaskProjector, self).__init__()
        self.input_dim = input_conv
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.name = name

        layers = []
        in_dim = self.input_dim
        if self.hidden_dims is None:
            layers.append(nn.Conv2d(in_dim, self.output_dim, kernel_size=3, padding=1))
        else:
            for hidden_dim in self.hidden_dims:
                layers += [nn.Conv2d(in_dim, hidden_dim, kernel_size=1, padding=0), nn.ReLU(inplace=True)]
                in_dim = hidden_dim
            layers.append(nn.Conv2d(in_dim, self.output_dim, kernel_size=3, padding=1))
        self.mlp_conv = nn.Sequential(*layers)
        
        # Initialization
        for m in self.mlp_conv.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """ Forward pass through the MLP projector.
        """
        return self.mlp_conv(x)
    
    def save_pretrained(self, save_directory):
        save_path = os.path.join(save_directory, f"{self.name}.bin")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.state_dict(), save_path)
        return f"Model saved to {save_path}"
    
    def from_pretrained(self, load_directory):
        load_path = os.path.join(load_directory, f"{self.name}.bin")
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"No model found at {load_path}")
        weight_dev = next(self.parameters()).device
        self.load_state_dict(torch.load(load_path, map_location=weight_dev))
        return f"Model loaded from {load_path}"