"""
Helperfunction for models
"""
 
def build_model(config):
    model_config = config["model"]
    model_name = model_config["name"].lower()
    params = model_config["params"]
    print("params in build model: ", params)

    if model_name == "transformer_style_mlp":
        from .transformer_style_mlp import TransformerStyleMLP   
        return TransformerStyleMLP(**params)
    elif model_name == "residual_mlp":
        from .residual_mlp import ResidualMLP   
        return ResidualMLP(**params)
    else:
        raise ValueError(f"Unsupported model name: {model_name}") 