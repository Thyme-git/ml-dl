import torch
from torch import nn
from einops.layers.torch import Rearrange
from einops import repeat

class FeedForward(nn.Module):
    """Feed forward network(ffn) after self-attention
        implemented as MLP with one hidden layer
    """
    def __init__(
            self,
            dim: int,
            hidden_dim:int,
            dropout: float = 0.1,
        ) -> None:
        """init function of ffn

        Args:
            dim (int): patch feature dim size
            hidden_dim (int): hidden size of hidden layer
        """
        
        super().__init__()
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): shape(b, n, dim)
        """
        return self.ffn(x)


class Attention(nn.Module):
    """
    multi-head self-attention module used in transformer
    """
    def __init__(
            self,
            dim: int,
            head_num: int,
            head_dim: int,
            dropout: float = 0.1,
        ) -> None:
        """
        Args:
            dim (int): patch feat dim
            head_num (int): attention heads num
            head_dim (int): attention heads dim
        """
        super().__init__()

        self.norm = nn.LayerNorm(dim)

        self.transform_qkv = nn.Linear(dim, head_dim*head_num*3, bias=False)
        self.split_heads = Rearrange("b n (h d) -> b h n d", d=head_dim)
        self.scoreScale = head_dim ** -0.5
        self.softmax = nn.Softmax(-1)
        self.concat_heads = nn.Sequential(
            Rearrange("b h n d -> b n (h d)"),
            nn.Linear(head_dim*head_num, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): (b, n, dim)
        """
        x = self.norm(x)

        qkv = self.transform_qkv(x)
        q, k, v = map(self.split_heads, torch.chunk(qkv, 3, dim = -1))

        attnScore = (q @ k.transpose(-1, -2)) *self.scoreScale
        attnWeight = self.softmax(attnScore) # (b, h, n, n)
        attn = attnWeight @ v # (b, h, n, hd)

        return self.concat_heads(attn)


class Transformer(nn.Module):
    """
    simple impletmentation of transformer
    """
    def __init__(
            self,
            dim: int,
            head_num: int,
            head_dim: int,
            ffn_hidden: int,
            layer_num: int,
            dropout: float = 0.1,
        ) -> None:
        """
        Args:
            dim (int): patch feature dim (input feat length)
            head_num (int): heads num of multi-head attention
            head_dim (int): size of each head
            ffn_hidden (int): hidden size of 2-layer MLP of feedforward network
            layer_num (int): layer(attention & ffn)'s num
            dropout (float, optional): dropout probability. Defaults to 0.1.
        """
        super().__init__()

        self.blocks = nn.ModuleList([])
        for _ in range(layer_num):
            self.blocks.append(nn.ModuleList([
                Attention(dim, head_num, head_dim, dropout),
                FeedForward(dim, ffn_hidden, dropout),
            ]))

    def forward(
            self,
            x: torch.Tensor,
        ):
        """
        Args:
            x (torch.Tensor): size(b, n, dim)
        """
        for attn, ffn in self.blocks:
            x = attn(x) + x
            x = ffn(x) + x

        return x


def make_pair(x: int| list):
    return [x, x] if isinstance(x, int) else x


class ViT(nn.Module):
    """
    Vision Transformer
    """
    def __init__(
            self,
            image_size: list,
            patch_size: int | list,
            dim: int,
            head_num: int,
            head_dim: int,
            ffn_hidden: int,
            layer_num:int,
            class_num: int,
            dropout:float = 0.1,
            channels: int = 3,
            feat_extract: str = "class"
        ) -> None:
        """
        Args:
            image_size (list): input image shape,
            patch_size (int | list): patch size,
            dim (int): linear projection dim,
            head_num (int): head num of multi-head self-attention,
            head_dim (int): head vector dim of multi-head self-attention,
            ffn_hidden (int): fead forward network(2 layer MLP) hidden size,
            layer_num (int): transformer layer num,
            class_num (int): class num corresponsed to your tasks,
            dropout:float = 0.1,
            channels: int = 3,
            feat_extract: str = "class"
        """

        super().__init__()
        
        # patch height, patch width
        self.ph, self.pw = make_pair(patch_size)
        self.h, self.w = image_size
        self.patch_num = self.h // self.ph * self.w // self.pw
        assert self.h % self.ph == 0 and self.w % self.pw == 0, f"patch size{[self.ph, self.pw]} not compatible with image size{[self.h, self.w]}"

        self.feat_extract = feat_extract

        self.split2patch = Rearrange("b c (nh ph) (nw pw) -> b (nh nw) (ph pw c)", ph=self.ph, pw=self.pw)

        self.patchProjection = nn.Linear(self.ph*self.pw*channels, dim)
        
        self.pos_token = nn.Parameter(torch.randn((1, self.patch_num+1, dim)))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(dropout)

        self.transformer = Transformer(dim, head_num, head_dim, ffn_hidden, layer_num, dropout)

        self.mlpHead = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, class_num),
            nn.Dropout(dropout),
        )


    def forward(
            self,
            x: torch.Tensor,
        ):
        """
        Args:
            x (torch.Tensor): size(b, c, h, w)
        """

        b, c, h, w = x.shape
        assert h % self.ph == 0 and w % self.pw == 0, f"patch size{[self.ph, self.pw]} not compatible with image size{[h, w]}"
        n = h // self.ph * w // self.pw

        x = self.split2patch(x)
        x = self.patchProjection(x)
        
        cls_token = repeat(self.cls_token, "1 1 d -> b 1 d", b=b)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_token[:, :(n+1)]
        x = self.dropout(x)

        x = self.transformer(x) # (b, n+1, dim)
        
        x = x[:, 0] if self.feat_extract == "class" else x.mean(dim=1)

        return self.mlpHead(x)