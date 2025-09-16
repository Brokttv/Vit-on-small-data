#

from torch import nn

class PatchEmbedding(nn.Module):
    """
    Reduces a 32x32 image with three stride-2 AvgPool2d ops:
    32 -> 16 -> 8 -> 4, then projects to embedding_dim.
    num_patches = (img_size//8) * (img_size//8).
    """
    def __init__(self,
                 in_channels: int = 3,

                 embedding_dim: int = 192,
                 img_size: int = 32):
        super().__init__()
        assert img_size % 8 == 0, \
            f"img_size {img_size} must be divisible by 8 due to three stride-2 pools."


        self.patcher = nn.Sequential(

    # Block 1: 32x32 -> 16x16
    nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
    nn.BatchNorm2d(64),
    nn.GELU(),
    nn.Conv2d(64, 64, kernel_size=3, padding=1),
    nn.BatchNorm2d(64),
    nn.GELU(),
    nn.AvgPool2d(kernel_size=2, stride=2),

    # Block 2: 16x16 -> 8x8
    nn.Conv2d(64, 128, kernel_size=3, padding=1),
    nn.BatchNorm2d(128),
    nn.GELU(),
    nn.Conv2d(128, 128, kernel_size=3, padding=1),
    nn.BatchNorm2d(128),
    nn.GELU(),
    nn.AvgPool2d(kernel_size=2, stride=2),

    # Block 3: 8x8 -> 4x4
    nn.Conv2d(128, 256, kernel_size=3, padding=1),
    nn.BatchNorm2d(256),
    nn.GELU(),
    nn.Conv2d(256, 256, kernel_size=3, padding=1),
    nn.BatchNorm2d(256),
    nn.GELU(),
    nn.AvgPool2d(kernel_size=2, stride=2),

    # Project to embedding_dim, keep HxW (now 4x4 for img_size=32)
    nn.Conv2d(256, embedding_dim, kernel_size=1),
    nn.BatchNorm2d(embedding_dim),
    nn.GELU()
)


        self.grid_h = img_size // 8
        self.grid_w = img_size // 8
        self.num_patches = self.grid_h * self.grid_w  
        self.flatten = nn.Flatten(start_dim=2, end_dim=3)  

    def forward(self, x):
        x = self.patcher(x)            
        x = self.flatten(x)              
        x = x.permute(0, 2, 1)           
        return x






class MultiHeadSelfAttentionBlock(nn.Module):
  def __init__(self,embedding_dim:int=192,num_heads:int=2,attention_dropout:float=0.0):
    super().__init__()

    self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)
    self.MSA = nn.MultiheadAttention(embed_dim=embedding_dim,num_heads=num_heads,dropout=attention_dropout, batch_first=True)

  def forward(self,x):
    x= self.layer_norm(x)
    att_output, _ = self.MSA(query=x,
                             key=x,
                             value=x.contiguous(),
                             need_weights=False)

    return att_output


class MLPBlock(nn.Module):
  def __init__(self,embedding_dim:int=192,mlp_size:int=1152,dropout:float=0.1):

    super().__init__()

    self.layer_norm = nn.LayerNorm(normalized_shape= embedding_dim)
    self.mlp = nn.Sequential(
    # First hidden layer
    nn.Linear(in_features=embedding_dim,
              out_features=mlp_size),
   nn.GELU(),
    nn.Dropout(p=dropout),



    # Output layer
    nn.Linear(in_features=mlp_size,
              out_features=embedding_dim),
    nn.Dropout(p=dropout)
)



  def forward(self,x):
    x = self.layer_norm(x)
    x= self.mlp(x)

    return x



class TransformerEncoderBlock(nn.Module):
  def __init__(self, embedding_dim:int=192, num_heads:int=2,mlp_size:int=1152, mlp_dropout:float=0.1, att_dropout:float=0.0):
    super().__init__()

    self.MSAblock = MultiHeadSelfAttentionBlock(embedding_dim = embedding_dim,num_heads=num_heads, attention_dropout = att_dropout)
    self.MLPblock= MLPBlock(embedding_dim=embedding_dim , mlp_size = mlp_size, dropout = mlp_dropout)

  def forward(self,x):
    x= x + self.MSAblock(x)
    x= x+ self.MLPblock(x)

    return x



class VIT(nn.Module):
  def __init__(self,in_channels:int=3,embedding_dim:int=192,num_heads:int=2,num_transformer_layers:int=2,attention_dropout:float=0.0,mlp_size:int=1152,mlp_dropout:float=0.1,embedding_dropout:float=0.1,num_classes:int=10, img_size:int=32):
    super().__init__()

    assert embedding_dim % num_heads == 0, \
            f"embedding_dim ({embedding_dim}) must be divisible by num_heads ({num_heads})."

        # Patch embedding via CNN pyramid 
    self.patch_embedding = PatchEmbedding(in_channels=in_channels,
                                              embedding_dim=embedding_dim,img_size=img_size)

        # 4. Calculate number of patches (height * width/patch^2)
    self.num_patches = self.patch_embedding.num_patches

        # 5. Create learnable class embedding 
    self.class_embedding = nn.Parameter(data=torch.randn(1, 1, embedding_dim),
                                            requires_grad=True)

        # 6. Create learnable position embedding
    self.position_embedding = nn.Parameter(data=torch.randn(1, self.num_patches+1, embedding_dim),
                                               requires_grad=True)

        # 7. Create embedding dropout value
    self.embedding_dropout = nn.Dropout(p=embedding_dropout)



    self.transformer_encoder = nn.Sequential(*[TransformerEncoderBlock(embedding_dim=embedding_dim, num_heads=num_heads,mlp_size=mlp_size, mlp_dropout=mlp_dropout, att_dropout=attention_dropout) for _ in range(num_transformer_layers)])
    self.classifier = nn.Sequential(nn.LayerNorm(normalized_shape=embedding_dim), nn.Linear(in_features=embedding_dim,out_features=num_classes))

  def forward(self,x):
   batch_size = x.shape[0]
   class_token = self.class_embedding.expand(batch_size,-1,-1)
   x = self.patch_embedding(x)
   x= torch.cat((class_token,x),dim=1)
   x= self.position_embedding.to(x.device) + x
   x= self.embedding_dropout(x)
   x= self.transformer_encoder(x)
   x= self.classifier(x[:,0])

   return x


model = VIT().to(device)
model
