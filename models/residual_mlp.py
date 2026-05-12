import torch.nn as nn
from utils.model_building_utils import get_activationfn

class ResidualMLPBlock(nn.Module):
    """
    Residual MLP Block: 
    x -> Linear(expansion) -> Activation -> Linear -> + residual.
    """

    def __init__(self, d_model, expansion,activation="gelu", **kwargs):
        super().__init__()
     
        act_layer = get_activationfn(activation)

        self.mlpblock = nn.Sequential(
            nn.Linear(d_model, expansion*d_model), 
            act_layer, 
            nn.Linear(expansion*d_model,d_model)
         )

    def forward(self,x):
        return self.mlpblock(x) + x
        

class ResidualMLP(nn.Module):
    """
    Definig residual MLP: A transformer like MLP without attention nor Layernorm.  
    """
    def __init__(self, d_model,  n_blocks, input_dim, output_dim,expansion=4, activation = "gelu", **kwargs):
        super().__init__()

        self.input_layer = nn.Linear(input_dim, d_model)

        self.mlp_blocks = nn.ModuleList(
            [
            ResidualMLPBlock(
                activation=activation, 
                d_model=d_model, 
                expansion=expansion)
                for _ in range(n_blocks)
            ]
        )
        self.output_layer = nn.Linear(d_model, output_dim)

    def forward(self, x):
        x = self.input_layer(x)

        for block in self.mlp_blocks:
            x = block(x)
        
        x = self.output_layer(x)
        return x