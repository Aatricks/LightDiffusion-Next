import os, sys
# Ensure project root is on PYTHONPATH like pytest does
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
from src.NeuralNetwork.flux2.model import Flux2

print('creating model (float32)')
model = Flux2(dtype=torch.float32)

vae_latent_small = torch.randn(1, 32, 96, 38)
t = torch.tensor([0.5])
txt = torch.zeros(1, 1, model.params.context_in_dim)
print('calling apply_model with transformer_options img_h=512,img_w=512')
try:
    out = model.apply_model(vae_latent_small, t, c_crossattn=txt, transformer_options={"img_h":512, "img_w":512})
    print('success; out.shape=', out.shape)
except Exception as e:
    import traceback
    traceback.print_exc()
    print('FAILED with exception')
