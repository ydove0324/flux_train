from diffusers import FluxTransformer2DModel
import torch

model = FluxTransformer2DModel.from_pretrained('black-forest-labs/FLUX.1-dev', subfolder='transformer')

# 获取包含attention的模块名称
attn_layers = [n for n, _ in model.named_modules() if 'attn' in n.lower()]
print("Attention layers:", attn_layers)

# 获取模型的前几层结构，以便观察命名模式
for name, _ in list(model.named_modules())[:20]:
    print(name)

# 查看第一个transformer_block的结构
for name, _ in model.named_modules():
    if 'transformer_blocks.0' in name:
        print(name) 