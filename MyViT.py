import torch
from torch import nn as nn

from Attention import MyMSA, get_positional_embeddings


class MyViT(nn.Module):

    def __init__(self, input_shape, n_patches=7, hidden_d=8, n_heads=2, out_d=10):
        super(MyViT, self).__init__()

        self.input_shape = input_shape  # Input shape is w.r.t. images -> (N, C, H, W)
        self.n_patches = n_patches  # We are breaking the images in n_patches x n_patches
        self.patch_size = (input_shape[1] / n_patches, input_shape[2] / n_patches)
        self.hidden_d = hidden_d

        # Check that the image is divisible entirely by the number of patches (both in width and height)
        assert input_shape[1] % n_patches == 0, "Input shape not entirely divisible by number of patches"
        assert input_shape[2] % n_patches == 0, "Input shape not entirely divisible by number of patches"

        # 1) Linear map
        self.input_d = int(input_shape[0] * self.patch_size[0] * self.patch_size[1])
        self.linear_mapper = nn.Linear(self.input_d, self.hidden_d)

        # 2) Classification token
        self.class_token = nn.Parameter(torch.rand(1, self.hidden_d))

        # 3) Positional embedding
        # (In forward method)

        # 4a) Layer normalization 1
        self.ln1 = nn.LayerNorm([self.n_patches ** 2 + 1, self.hidden_d])

        # 4b) Multi-head Self Attention (MSA) and classification token
        self.msa = MyMSA(self.hidden_d, n_heads)

        # 5a) Layer Normalization 2
        self.ln2 = nn.LayerNorm([self.n_patches ** 2 + 1, self.hidden_d])

        # 5b) Encoder MLP
        self.enc_mlp = nn.Sequential(
            nn.Linear(self.hidden_d, self.hidden_d),
            nn.ReLU()
        )

        # 6) Classification MLP
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_d, out_d),
            nn.Softmax(dim=-1)
        )

    def forward(self, images):
        # Dividing images into patches
        n, c, w, h = images.shape
        patches = images.reshape(n, self.n_patches ** 2, self.input_d)

        # Running linear layer for tokenization
        tokens = self.linear_mapper(patches)

        # Adding classification token to the tokens
        tokens = torch.stack([torch.vstack((self.class_token, tokens[i])) for i in range(len(tokens))])

        # Adding positional embedding
        tokens += get_positional_embeddings(self.n_patches ** 2 + 1, self.hidden_d).repeat(n, 1, 1).to(self.device())

        # Transformer
        out = tokens + self.msa(self.ln1(tokens))
        out = out + self.enc_mlp(self.ln2(out))

        # Get the classification token only
        out = out[:, 0]

        return self.mlp(out)

    def device(self):
        return next(self.parameters()).device
