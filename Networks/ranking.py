






class SelectionUNet(nn.Module):
    # input is strict to 96, 96, 96
    
    def __init__(self, img_shape, token_length, encoder_drop, transformer_drop):
        super().__init__()
        W, H, D = img_shape
        
        self.encoder = SwinTransformer(
            in_chans = 2,
            embed_dim = 24,
            window_size = (7, 7, 7),
            patch_size= (2, 2, 2),
            depths = (2, 2, 2, 2),
            num_heads = (3, 6, 12, 24),
            attn_drop_rate=encoder_drop,
            drop_path_rate=encoder_drop
        )
        self.convert_x_24 = nn.Sequential(
            nn.Conv3d(24, 24, (3, 3, 3), (2, 2, 2), (1, 1, 1)),
            nn.Conv3d(24, 24, (3, 3, 3), (2, 2, 2), (1, 1, 1)),
            nn.Conv3d(24, 24, (3, 3, 3), (2, 2, 2), (1, 1, 1)),
            nn.Conv3d(24, 24, (3, 3, 3), (2, 2, 2), (1, 1, 1)),
            nn.Conv3d(24, token_length, (3, 3, 2)),
            nn.Flatten(1),
            nn.ReLU()
        )
        self.convert_x_48 = nn.Sequential(
            nn.Conv3d(48, 48, (3, 3, 3), (2, 2, 2), (1, 1, 1)),
            nn.Conv3d(48, 48, (3, 3, 3), (2, 2, 2), (1, 1, 1)),
            nn.Conv3d(48, 48, (3, 3, 3), (2, 2, 2), (1, 1, 1)),
            nn.Conv3d(48, token_length, (3, 3, 2)),
            nn.Flatten(1),
            nn.ReLU()
        )
        self.convert_x_96 = nn.Sequential(
            nn.Conv3d(96, 96, (3, 3, 3), (2, 2, 2), (1, 1, 1)),
            nn.Conv3d(96, 96, (3, 3, 3), (2, 2, 2), (1, 1, 1)),
            nn.Conv3d(96, token_length, (3, 3, 2)),
            nn.Flatten(1),
            nn.ReLU()
        )
        self.convert_x_192 = nn.Sequential(
            nn.Conv3d(192, 192, (3, 3, 3), (2, 2, 2), (1, 1, 1)),
            nn.Conv3d(192, token_length, (3, 3, 2)),
            nn.Flatten(1),
            nn.ReLU()
        )
        self.convert_x_384 = nn.Sequential(
            nn.Conv3d(384, token_length, (3, 3, 2)),
            nn.Flatten(1),
            nn.ReLU()
        )
        self.projection_individual = nn.Sequential(
            nn.Conv3d(384, token_length, (4, 4, 2), 1, 0),
            nn.Flatten(start_dim=1),
            # nn.Linear(token_length, token_length),
            nn.Sigmoid()
        )
        # self.softmax = torch.nn.Softmax(dim=0)
        self.tf_4 = TransformerBlock(token_length, token_length + 2048, 8, transformer_drop)
        self.tf_3 = TransformerBlock(token_length, token_length+ 2048, 8, transformer_drop)
        self.tf_2 = TransformerBlock(token_length, token_length+ 2048, 8, transformer_drop)
        self.tf_1 = TransformerBlock(token_length, token_length+ 2048, 8, transformer_drop)
        self.tf_0 = TransformerBlock(token_length, token_length+ 2048, 8, transformer_drop)
        
        self.bn_4 = nn.BatchNorm1d(token_length)
        self.bn_3 = nn.BatchNorm1d(token_length)
        self.bn_2 = nn.BatchNorm1d(token_length)
        self.bn_1 = nn.BatchNorm1d(token_length)
        self.bn_0 = nn.BatchNorm1d(token_length)



        self.projection_tf = nn.Sequential(
            nn.Linear(token_length, 1),
            nn.Softmax(dim=0)
        )



    def forward(self, x):
        
        x = self.encoder(x)
        
        
        x_24 = x[0]
        x_24 = self.convert_x_24(x_24)
        x_24 = self.bn_4(x_24).unsqueeze(0)
        
        x_48 = x[1]
        x_48 = self.convert_x_48(x_48)
        x_48 = self.bn_3(x_48).unsqueeze(0)
        
        x_96 = x[2]
        x_96 = self.convert_x_96(x_96)
        x_96 = self.bn_2(x_96).unsqueeze(0)
        
        x_192 = x[3]
        x_192 = self.convert_x_192(x_192)
        x_192 = self.bn_1(x_192).unsqueeze(0)
        
        x_384 = x[4]
        x_384 = self.convert_x_384(x_384)
        x_384 = self.bn_0(x_384).unsqueeze(0)



        x = self.tf_4(x_384)
        
        x = self.tf_3(x + x_192)
        
        x = self.tf_2(x + x_96)

        x = self.tf_1(x + x_48)
        
        x = self.tf_0(x + x_24)
        

        
        return self.projection_tf(x.squeeze(0))