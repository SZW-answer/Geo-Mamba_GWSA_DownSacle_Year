# Nets.py
import torch
import torch.nn as nn
import math
from efficient_kan import KAN, KANLinear
import Debug
from mamba_ssm import Mamba
class AdaptivePositionEncoder(nn.Module):
    """
    自适应位置编码器，输入为纬度和经度，输出为位置嵌入。
    具备距离保持性和方向感知性。
    """
    def __init__(self, input_dim=2, embed_dim=64):
        super(AdaptivePositionEncoder, self).__init__()
        self.position_embedding = nn.Sequential(
            KANLinear(input_dim, embed_dim*2),
            # nn.ReLU(),
            KANLinear(embed_dim*2, embed_dim),
            # nn.ReLU(),
        )
        self.RMSNorm =nn.RMSNorm(embed_dim)
        self.RMSNorm2=nn.RMSNorm(2)
        # self._initialize_rmsnorm()
    def _initialize_rmsnorm(self):
        """
        自定义 RMSNorm 层的初始化方法。
        """
        # 初始化 RMSNorm 的权重（如果有的话）
        if hasattr(self.RMSNorm, 'weight') and self.RMSNorm.weight is not None:
            nn.init.ones_(self.RMSNorm.weight)  # 或者其他合适的初始化方法
        # 如果 RMSNorm 有偏置项，也可以初始化
        if hasattr(self.RMSNorm, 'bias') and self.RMSNorm.bias is not None:
            nn.init.zeros_(self.RMSNorm.bias)  
    def forward(self, lat, lon):
        """
        lat: [batch_size, 1]
        lon: [batch_size, 1]
        """
        position = torch.cat([lat, lon], dim=-1)  # [batch_size, 2]
        position = self.RMSNorm2(position)
        pos_embed = self.position_embedding(position)  # [batch_size, embed_dim]
        pos_embed = self.RMSNorm(pos_embed)
        return pos_embed  # [batch_size, embed_dim]

class StaticFeatureEncoder(nn.Module):
    """
    静态特征编码器，处理 DEM 和 aspect 等静态变量。
    """
    def __init__(self, static_dim, embed_dim=64):
        super(StaticFeatureEncoder, self).__init__()
        self.static_encoder = nn.Sequential(
            KANLinear(static_dim, embed_dim*2),
            # nn.ReLU(),
            KANLinear(embed_dim*2, embed_dim),
            # nn.ReLU(),
        )
        self.RMSNorm = nn.RMSNorm(static_dim)

    def forward(self, static_features):
        """
        static_features: [batch_size, static_dim]
        """
        static_features = self.RMSNorm(static_features)
        static_embed = self.static_encoder(static_features)  # [batch_size, embed_dim]
        return static_embed  # [batch_size, embed_dim]
class MambaEncoder(nn.Module):
    def __init__(self, num_layers, mambadim, d_state=16, d_conv=4, expand=2):
        super(MambaEncoder, self).__init__()
        self.layers = nn.ModuleList([
            Mamba(
                d_model=mambadim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand
            ) for _ in range(num_layers)
        ])
        self.layer_norm = nn.RMSNorm(mambadim)
        self.GELU = nn.GELU()
        self.MLP = nn.Sequential(
            # KANLinear(mambadim,256),
            nn.Linear(mambadim,1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024,mambadim),
            nn.Dropout(0.1),
            nn.GELU()
        )
    def forward(self, x):
        for layer in self.layers:
            residual = x
            x = layer(x)
            # x = x + residual  # 残差连接
            # residual = x
            # x = self.layer_norm(x)
            self.MLP(x)
            x = x + residual  # 残差连接
            x = self.layer_norm(x)
            # x = self.GELU(x)
        return x

class CrossAttentionModule(nn.Module):
    """
    交叉注意力模块，用于融合不同种类的嵌入。
    """
    def __init__(self, embed_dim=64, num_heads=4):
        super(CrossAttentionModule, self).__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.layer_norm = nn.RMSNorm(embed_dim)
        self.feed_forward = nn.Sequential(
            KANLinear(embed_dim, embed_dim * 2),
            nn.GELU(),
            KANLinear(embed_dim * 2, embed_dim),
            nn.GELU(),
            KANLinear(embed_dim,embed_dim),
            nn.GELU()

        )
        self.ffn_norm = nn.RMSNorm(embed_dim)

    def forward(self, query, key, value):
        """
        query, key, value: [batch_size, seq_len, embed_dim]
        """
        attn_output, _ = self.cross_attn(query, key, value)  # attn_output: [batch_size, seq_len, embed_dim]
        out = self.layer_norm(query + attn_output)
        ff_output = self.feed_forward(out)
        out = self.ffn_norm(out + ff_output)
        return out  # [batch_size, seq_len, embed_dim]



class FeatureMamba(nn.Module):
    def __init__(self, embed_dim, mambadim,num_layers,d_state,d_conv,expand):
        super(FeatureMamba, self).__init__()
        self.tmpembed = KANLinear(1, mambadim)
        self.tmpembed_t = KANLinear(mambadim, 1)
        self.mamba_encoder = MambaEncoder(
            num_layers=num_layers,
            mambadim=mambadim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        self.LayerNorm = nn.RMSNorm(mambadim)
        self.LayerNorm_x= nn.RMSNorm(embed_dim)
    def forward(self, x):
        residual = x  # 残差连接
        tmp = x.view(x.size(0), x.size(2), x.size(1))  # B 20,1
        tmp = self.tmpembed(tmp)  # B，20,20 64
        dynamic_encoded = self.mamba_encoder(tmp)
        dynamic_encoded = self.tmpembed_t(dynamic_encoded)
        dynamic_encoded = dynamic_encoded.view(x.size(0), x.size(1), x.size(2))
        return self.LayerNorm_x(dynamic_encoded + residual)  # 加上残差连接


class TransformerRegressor(nn.Module):
    """
    编码器-解码器结构的Transformer回归模型，用于预测GWS变量。
    """
    def __init__(self,  LC_num_classes,
                 static_dim=3, dynamic_dim=16, embed_dim=64, num_heads=4,
                 num_layers=2, seq_len=10):
        super(TransformerRegressor, self).__init__()
        self.seq_len = seq_len
        self.LC_embedding =nn.Sequential(
            nn.Embedding(LC_num_classes, 1),
        )
        
        nn.Embedding(LC_num_classes, 1)
        self.all_feat_dim = static_dim+dynamic_dim+1
        self.mambadim = 512
        self.LayerNorm = nn.RMSNorm(self.mambadim)
        self.LayerNorm_embedim = nn.RMSNorm(self.mambadim)
        self.LayerNorm_embedim2 = nn.RMSNorm(embed_dim)
        # 自适应位置编码器
        self.position_encoder = AdaptivePositionEncoder(input_dim=2, embed_dim=embed_dim)

        # 静态特征编码器 (DEM, aspect)
        self.static_encoder = StaticFeatureEncoder(static_dim=static_dim, embed_dim=embed_dim)

        # 交叉注意力模块
        self.cross_attn_all1 = CrossAttentionModule(embed_dim=embed_dim, num_heads=num_heads)
        self.cross_attn_all2 = CrossAttentionModule(embed_dim=embed_dim, num_heads=num_heads)

        # Transformer Encoder
        # encoder_layer = nn.TransformerEncoderLayer(d_model=self.mambadim, nhead=num_heads, activation='relu', batch_first=True)
        # self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # self.LC_embedding = nn.Embedding(LC_num_classes, embed_dim)
        # 最终回归层
        self.final_layernorm = nn.RMSNorm(self.all_feat_dim*2)
        self.regressor2 = nn.Sequential(
            KANLinear(self.all_feat_dim, embed_dim*2 ),
            KANLinear( embed_dim*2 ,self.all_feat_dim),
            KANLinear(self.all_feat_dim,1)
        )
        self.mamba_encoder_64 = FeatureMamba(
            num_layers=1,
            mambadim=self.mambadim,
            d_state=16,
            d_conv=4,
            expand=2,
            embed_dim=self.all_feat_dim-1
        )
        self.mamba_encoder_final_64 = FeatureMamba(
            num_layers=num_layers,
            mambadim=self.mambadim,
            d_state=16,
            d_conv=4,
            expand=2,
            embed_dim=self.all_feat_dim
        )
        self.mamba_encoder1_64 = FeatureMamba(
            num_layers=1,
            mambadim=self.mambadim,
            d_state=16,
            d_conv=4,
            expand=2,
            embed_dim=embed_dim
        )
        self.mamba_encoder2_64 = FeatureMamba(
            num_layers=1,
            mambadim=self.mambadim,
            d_state=16,
            d_conv=4,
            expand=2,
            embed_dim=embed_dim
        )
        self.mamba_encoder3_64 = FeatureMamba(
            num_layers=num_layers,
            mambadim=self.mambadim,
            d_state=16,
            d_conv=4,
            expand=2,
            embed_dim=embed_dim
        )
    
    
        self.dynamic_embed = nn.Sequential(
            KANLinear(self.all_feat_dim-1,embed_dim)
        )
        self.dynamic_embed_down = nn.Sequential(
            KANLinear(embed_dim,self.all_feat_dim)
        )
        
        self.dynamic_embed2 = nn.Sequential(
            KANLinear(self.all_feat_dim,embed_dim)
        )


        self.tmpembed = KANLinear(1,self.mambadim)
        self.tmpembed_t = KANLinear(self.mambadim,1)
        
        self.embdim_embed = KANLinear(1,self.mambadim)
        self.embdim_embed_t=KANLinear(self.mambadim,1)

        
        self.LC_Encoder= KANLinear(1,embed_dim)

    def forward(self, x, lat, lon, static_features):
        """
        x: [batch_size, seq_len, dynamic_dim] - 动态特征
        lat: [batch_size, 1] - 纬度
        lon: [batch_size, 1] - 经度
        static_features: [batch_size, static_dim] - 静态特征 (DEM, aspect)
        """
        #
        batch_size = x.size(0)
        LC  = x[:, :, -1].long()
        x= x[:,:,:-1]
        LC=self.LC_embedding(LC)
        LC = self.LC_Encoder(LC)
        # 获取位置嵌入 1024*64
        lat = lat.unsqueeze(-1) # [batch_size, seq_len, embed_dim]
        lon = lon.unsqueeze(-1)  # [batch_size, seq_len, embed_dim]
        pos_embed = self.position_encoder(lat, lon)  # [batch_size, embed_dim]
        # 取静态特征嵌入
        static_embed = self.static_encoder(static_features)  # [batch_size, embed_dim]
       
        # 编码动态特征
        # 编码动态特征，并融合位置嵌入
        x = torch.cat([x,static_features],dim=-1) # （b,1,20）
        dynamic_encoded = self.mamba_encoder_64(x)
        dynamic_encoded = self.dynamic_embed(dynamic_encoded) #20维度-》embeddim 升维度
        
    
        # 交叉注意力融合位置嵌入和静态嵌入到动态特征
        # 动态变量查找位置信息
        # 算自注意力
        #mamab搞一层 
        fused_pos_dy = self.cross_attn_all1(dynamic_encoded,LC,LC)
        fused_pos_dy = self.mamba_encoder1_64(fused_pos_dy)
        # 动态变量查找静态信息
        fused_pos_dy =self.cross_attn_all2(fused_pos_dy, static_embed,static_embed)
        fused_pos_dy = self.mamba_encoder2_64(fused_pos_dy)
        fused_pos_dy = self.cross_attn_all1(fused_pos_dy,pos_embed,pos_embed)
        fused_pos_dy = self.mamba_encoder3_64(fused_pos_dy)
        # 降维
        dynamic_encoded_down = self.dynamic_embed_down(fused_pos_dy)
        dynamic_encoded_down =self.mamba_encoder_final_64(dynamic_encoded_down)
       
        # 最终回归
        output = self.regressor2(dynamic_encoded_down)  # [batch_size, 1]
        output = output.squeeze(-1)
       
        return output  # [batch_size]
