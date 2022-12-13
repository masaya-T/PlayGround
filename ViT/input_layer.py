import torch
import torch.nn as nn 

class VitInputLayer:
    def __init__(self,
                 in_channels:int=3,
                 emb_dim:int=384,
                 num_patch_row:int=2,
                 image_size:int=32
                ) -> None:
        self.in_channels = in_channels
        self.emb_dim = emb_dim
        self.num_putch_row = num_patch_row
        self.image_size = image_size

        # パッチの数
        self.num_putch = self.num_putch_row ** 2
        
        # パッチの大きさ
        self.patch_size = int(self.image_size // self.num_putch_row)
        
        # 入力画像のパッチへの分割　＆　パッチの埋め込みを一気に行う層
        self.patch_emb_layer = nn.Conv2d(in_channels=self.in_channels,
                                         out_channels=self.emb_dim,
                                         kernel_size=self.patch_size,
                                         stride=self.patch_size)

        # クラストークン
        self.cls_token = nn.Parameter( torch.randn( 1, 1, emb_dim ))
        
        # 位置埋め込み
        self.pos_emb = nn.Parameter( torch.randn( 1, self.num_patch + 1, emb_dim ))
    
    def forward( self, x : torch.Tensor ) -> torch.Tensor:

        # パッチの埋め込み
        z_0 = self.patch_emb_layer( x )

        # パッチのflatten
        z_0 = z_0.flatten( 2 )

        # 軸の入れ替え
        z_0 = z_0.transpose(1,2)

        # パッチの埋め込みの先頭にクラストークンの結合
        z_0 = torch.cat( [self.cls_token.repeat( repeats = ( x.size(0), 1, 1)), z_0], dim=1 )

        # 一埋め込みの加算
        z_0 = z_0 + self.pos_emb

        return z_0
