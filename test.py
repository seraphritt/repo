
import streamlit as st
import torch, torch.nn.functional as F
from torchvision.utils import make_grid
from PIL import Image
import numpy as np

CKPT = "decoder.pt"
LATENT, IMG_SIZE = 20, 28

# --- same decoder arch as in train_cvae.py ---
class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(LATENT + 10, 400)
        self.fc2 = torch.nn.Linear(400, IMG_SIZE*IMG_SIZE)

    def forward(self, z, y_onehot):
        z = torch.cat([z, y_onehot], dim=1)
        h = torch.relu(self.fc1(z))
        return torch.sigmoid(self.fc2(h))

device = "cpu"  # good enough for inference
decoder = Decoder().to(device)
decoder.load_state_dict(torch.load(CKPT, map_location=device))
decoder.eval()

st.set_page_config(page_title="Handwritten Digit Image Generator", layout="centered")

st.title("Handwritten Digit Image Generator")

digit = st.selectbox("Choose a digit to generate (0-9)", list(range(10)), index=0)
if st.button("Generate 5 images"):
    with torch.no_grad():
        y = torch.tensor([digit]*5)
        y_one = F.one_hot(y, num_classes=10).float().to(device)
        z = torch.randn(5, LATENT).to(device)
        imgs = decoder(z, y_one).cpu().view(-1, 1, IMG_SIZE, IMG_SIZE)

    grid = make_grid(imgs, nrow=5, padding=5).mul(255).byte().permute(1,2,0).squeeze()
    st.image(Image.fromarray(grid.numpy()), caption=f"Generated images of digit {digit}", width=400)