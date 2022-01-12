from model.convnext import ConvNext

depths_dims = dict(
    convnext_tiny = (dict(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768])),
    convnext_small = (dict(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768])),
    convnext_base = (dict(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024])),  
    convnext_large = (dict(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536])),
    convnext_xlarge = (dict(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048])),
)

def build_model(name, **kwargs):
    model = ConvNext(**depths_dims[name], **kwargs)
    return model