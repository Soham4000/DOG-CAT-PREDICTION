# ============================================================
# 13. CONVERT TRAINED MODEL FOR STREAMLIT DEPLOYMENT (Python 3.14)
# ============================================================
# Everything above this line is exactly your original code, unchanged.
# This section only runs AFTER training finishes — it doesn't touch how
# the model is built or trained.
#
# Why this is needed: your deployed Streamlit app runs on Python 3.14,
# and TensorFlow has no build for 3.14, so the app can't load a .keras
# file directly. PyTorch does support 3.14, so this converts your
# trained model's weights into a PyTorch-loadable file — same
# architecture, same trained weights, just a different file format.
#
# Weights are saved as float16 instead of float32 to stay under 25MB —
# your Dense(512) layer alone has ~8.67M parameters (~35MB in float32).
# float16 halves that to ~17.4MB.
 
import torch
import os
 
conv_layers = [l for l in model.layers if 'conv2d' in l.name]
dense_layers = [l for l in model.layers if 'dense' in l.name]
 
state_dict = {}
 
for i, layer in enumerate(conv_layers, start=1):
    w, b = layer.get_weights()
    w_t = np.transpose(w, (3, 2, 0, 1))  # Keras (kh,kw,in,out) -> PyTorch (out,in,kh,kw)
    state_dict[f'conv{i}.weight'] = torch.tensor(w_t).half()
    state_dict[f'conv{i}.bias'] = torch.tensor(b).half()
 
for i, layer in enumerate(dense_layers, start=1):
    w, b = layer.get_weights()
    w_t = np.transpose(w, (1, 0))  # Keras (in,out) -> PyTorch (out,in)
    state_dict[f'fc{i}.weight'] = torch.tensor(w_t).half()
    state_dict[f'fc{i}.bias'] = torch.tensor(b).half()
 
torch.save(state_dict, 'dog_cat_model.pt')
 
size_mb = os.path.getsize('dog_cat_model.pt') / 1e6
print(f"Saved dog_cat_model.pt: {size_mb:.2f} MB")
print("Download this file (left sidebar -> Files -> right-click -> Download) "
      "and push it to your GitHub repo root.")
