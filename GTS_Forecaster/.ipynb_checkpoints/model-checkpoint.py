import torch
import torch.nn as nn
import torch.nn.functional as F
from fftKAN import *
from effKAN import *

# 1. Normalize 类
class Normalize(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True, subtract_last=False, non_norm=False):
        super(Normalize, self).__init__()
        self.subtract_last = subtract_last
        self.non_norm = non_norm
        if not self.non_norm:
            self.norm = nn.BatchNorm1d(num_features, affine=affine, eps=eps)

    def forward(self, x, mode='norm'):
        if self.non_norm:
            return x
        if mode == 'norm':
            # x 的形状: (batch, seq_len, num_features) -> (batch, num_features, seq_len)
            x = x.permute(0, 2, 1)
            x = self.norm(x)
            # 恢复形状
            x = x.permute(0, 2, 1)
            return x
        elif mode == 'denorm':
            # 反标准化
            # 这里假设 BatchNorm1d 没有逆操作，可以根据需求自定义反标准化
            raise NotImplementedError("Denormalization is not implemented.")
        else:
            raise ValueError("Mode should be 'norm' or 'denorm'")

# 2. DataEmbedding_wo_pos 类
class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.temporal_embedding = nn.Linear(10, d_model)  # 假设时间特征维度为10
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)

# 3. series_decomp 类
class series_decomp(nn.Module):
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.kernel_size = kernel_size
        self.moving_avg = nn.Conv1d(1, 1, kernel_size, padding=(kernel_size//2), bias=False)
        # 初始化移动平均滤波器权重为均值滤波器
        nn.init.constant_(self.moving_avg.weight, 1.0 / kernel_size)

    def forward(self, x):
        # x 的形状: (batch, channels, seq_len)
        moving_mean = self.moving_avg(x)
        trend = moving_mean
        seasonal = x - trend
        return trend, seasonal

# 4. DFT_series_decomp 类
class DFT_series_decomp(nn.Module):
    def __init__(self, top_k):
        super(DFT_series_decomp, self).__init__()
        self.top_k = top_k

    def forward(self, x):
        # x 的形状: (batch, channels, seq_len)
        fft = torch.fft.fft(x)
        fft[:, :, self.top_k:-self.top_k] = 0
        seasonal = torch.fft.ifft(fft).real
        trend = x - seasonal
        return trend, seasonal

# 5. MultiScaleSeasonMixing 类
class MultiScaleSeasonMixing(nn.Module):
    def __init__(self, configs):
        super(MultiScaleSeasonMixing, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(configs['mixing_layers']):
            self.layers.append(nn.Sequential(
                nn.Linear(configs['season_dim'], configs['season_dim']),
                nn.ReLU()
            ))

    def forward(self, seasonal):
        for layer in self.layers:
            seasonal = layer(seasonal)
        return seasonal

# 6. MultiScaleTrendMixing 类
class MultiScaleTrendMixing(nn.Module):
    def __init__(self, configs):
        super(MultiScaleTrendMixing, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(configs['mixing_layers']):
            self.layers.append(nn.Sequential(
                nn.Linear(configs['trend_dim'], configs['trend_dim']),
                nn.ReLU()
            ))

    def forward(self, trend):
        for layer in self.layers:
            trend = layer(trend)
        return trend

# 7. PastDecomposableMixing 类
class PastDecomposableMixing(nn.Module):
    def __init__(self, configs):
        super(PastDecomposableMixing, self).__init__()
        if configs['decomp_method'] == 'moving_avg':
            self.decomposition = series_decomp(configs['kernel_size'])
        elif configs['decomp_method'] == 'dft':
            self.decomposition = DFT_series_decomp(configs['top_k'])
        else:
            raise ValueError("Unsupported decomposition method")
        self.season_mixing = MultiScaleSeasonMixing(configs)
        self.trend_mixing = MultiScaleTrendMixing(configs)

    def forward(self, x):
        # x 的形状: (batch, seq_len, num_features)
        # 转换为 (batch, num_features, seq_len) 以适应 Conv1d
        x = x.permute(0, 2, 1)
        trend, seasonal = self.decomposition(x)
        # 转回 (batch, seq_len, num_features)
        trend = trend.permute(0, 2, 1)
        seasonal = seasonal.permute(0, 2, 1)
        seasonal = self.season_mixing(seasonal)
        trend = self.trend_mixing(trend)
        return trend + seasonal

# 8. TextModel 类
class TextModel(nn.Module):
    def __init__(self, config):
        super(TextModel, self).__init__()
        self.embedding = nn.Linear(config['c_in'], config['d_model'])
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config['d_model'],
            nhead=config['transformer_num_heads'],
            dropout=config['transformer_dropout']  # 修改为使用 'transformer_dropout'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config['num_encoder_layers'])
        self.fc_out = nn.Linear(config['d_model'], config['output_dim'])
        self.dropout = nn.Dropout(config['transformer_dropout'])  # 修改为使用 'transformer_dropout'

    def forward(self, x):
        # 假设 x 的形状为 (batch_size, seq_length, input_dim)
        x = self.embedding(x)  # (batch_size, seq_length, d_model)
        x = x.permute(1, 0, 2)  # Transformer 期望输入形状为 (seq_length, batch_size, d_model)
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)  # (batch_size, seq_length, d_model)
        x = self.dropout(x)
        x = self.fc_out(x[:, -1, :])  # 使用最后一个时间步的输出
        return x

# 9. LSTM 类
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim  # 隐藏层维度
        self.num_layers = num_layers  # LSTM层的数量
        # LSTM网络层
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        # 全连接层，用于将LSTM的输出转换为最终的输出维度
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        # 前向传播LSTM，返回输出和最新的隐藏状态与细胞状态
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        # 将LSTM的最后一个时间步的输出通过全连接层
        out = self.fc(out[:, -1, :])
        return out

class LSTM_Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout_rate=0.5):
        super(LSTM_Attention, self).__init__()
        self.hidden_dim = hidden_dim  # 隐藏层维度
        self.num_layers = num_layers  # LSTM层的数量
        self.dropout_rate = dropout_rate  # Dropout率

        # LSTM网络层
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_rate)
        # Dropout层
        self.dropout = nn.Dropout(dropout_rate)
        # 注意力机制
        self.attention = nn.Linear(hidden_dim, 1)
        # 全连接层，用于将LSTM的输出转换为最终的输出维度
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # 前向传播LSTM，返回输出和最新的隐藏状态与细胞状态
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        
        # 应用Dropout
        out = self.dropout(out)
        
        # 注意力机制
        attention_weights = F.softmax(self.attention(out), dim=1)
        context_vector = torch.sum(out * attention_weights, dim=1)
        
        # 将LSTM的输出通过全连接层
        out = self.fc(context_vector)
        return out

# 10. LSTM_ekan 类
class LSTM_ekan(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM_ekan, self).__init__()
        self.hidden_dim = hidden_dim  # 隐藏层维度
        self.num_layers = num_layers  # LSTM层的数量
        # LSTM网络层
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        # 全连接层，用于将LSTM的输出转换为最终的输出维度
        self.e_kan = KAN([hidden_dim, output_dim])

    def forward(self, x):
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        # 前向传播LSTM，返回输出和最新的隐藏状态与细胞状态
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        # 将LSTM的最后一个时间步的输出通过全连接层
        out = self.e_kan(out[:, -1, :])
        return out


class VanillaLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(VanillaLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # LSTM网络层
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        # 自定义全连接层
        self.fc = nn.Linear(hidden_dim, output_dim)

    def init_hidden(self, batch_size):
        # 初始化隐藏状态和细胞状态
        return (torch.zeros(self.num_layers, batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, batch_size, self.hidden_dim))

    def forward(self, x):
        # 初始化隐藏状态和细胞状态
        h0, c0 = self.init_hidden(x.size(0))
        # 前向传播LSTM
        out, _ = self.lstm(x, (h0, c0))
        # 将LSTM的最后一个时间步的输出通过自定义全连接层
        out = self.fc(out[:, -1, :])
        return out

class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, output_dim):
        super(ConvLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.output_dim = output_dim

        # 确保input_size与输入数据的特征维度相匹配
        self.conv_lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers,
                                 batch_first=True, bidirectional=False)

        # 卷积层，用于将卷积LSTM的输出转换为最终的输出维度
        self.conv = nn.Conv1d(hidden_dim, output_dim, kernel_size=kernel_size, padding=(kernel_size-1)//2)

    def forward(self, x):
        # x的形状: (batch, seq_len, num_features)
        # 转换为 (batch, num_features, seq_len) 以适应 Conv1d
        x = x.permute(0, 2, 1)
        
        # 前向传播ConvLSTM
        out, _ = self.conv_lstm(x)
        
        # 将ConvLSTM的输出通过卷积层
        out = out.permute(0, 2, 1)  # 转换回 (batch, seq_len, num_features)
        out = self.conv(out)
        
        # 取每个序列的最后一个输出
        out = out[:, :, -1]  # (batch, output_dim)
        
        return out

class CustomLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(CustomLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 自定义 LSTM 单元，使用 ReLU 激活函数
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        # 前向传播 LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_dim)

        # 应用 ReLU 激活函数
        out = self.relu(out)

        # 取最后一个时间步的输出
        out = out[:, -1, :]  # (batch_size, hidden_dim)

        # 全连接层
        out = self.fc(out)
        return out

class ConvGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, output_dim):
        super(ConvGRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.output_dim = output_dim

        # 输入层卷积
        self.conv_input = nn.Conv1d(input_dim, hidden_dim, kernel_size, padding=(kernel_size-1)//2)
        
        # 堆叠ConvGRU层
        self.conv_gru_layers = nn.ModuleList([ConvGRULayer(hidden_dim, hidden_dim, kernel_size) for _ in range(num_layers)])
        
        # 输出层
        self.fc_output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x的形状: (batch, seq_len, num_features)
        h = self.conv_input(x.transpose(1, 2))  # 转换为(batch, num_features, seq_len)并应用卷积
        for gru_layer in self.conv_gru_layers:
            h, _ = gru_layer(h, None)  # ConvGRULayer期望一个初始隐藏状态，这里我们传递None
        output, _ = h.max(dim=2)  # 取序列的最大值作为输出
        output = self.fc_output(output)
        return output

class ConvGRULayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size):
        super(ConvGRULayer, self).__init__()
        self.hidden_dim = hidden_dim  # 添加 hidden_dim 属性
        self.update_gate = nn.Conv1d(input_dim + hidden_dim, hidden_dim, kernel_size, padding=(kernel_size-1)//2)
        self.reset_gate = nn.Conv1d(input_dim + hidden_dim, hidden_dim, kernel_size, padding=(kernel_size-1)//2)
        self.tanh_part = nn.Conv1d(input_dim + hidden_dim, hidden_dim, kernel_size, padding=(kernel_size-1)//2)

    def forward(self, x, h_prev):
        if h_prev is None:
            h_prev = torch.zeros(x.size(0), self.hidden_dim, x.size(2)).to(x.device)
        combined = torch.cat([x, h_prev], dim=1)
        z = torch.sigmoid(self.update_gate(combined))
        r = torch.sigmoid(self.reset_gate(combined))
        h_hat = torch.tanh(self.tanh_part(torch.cat([x, r * h_prev], dim=1)))
        h_next = (1 - z) * h_prev + z * h_hat
        return h_next, h_next

# 11. GRU 类
class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(GRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # GRU网络层
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        # 全连接层，与LSTM相同
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        # 前向传播GRU
        out, hn = self.gru(x, h0.detach())
        # 将GRU的最后一个时间步的输出通过全连接层
        out = self.fc(out[:, -1, :])
        return out

class OptimizedGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, output_dim=1, 
                 bidirectional=True, dropout=0.3, use_layer_norm=True):
        """
        优化版GRU实现
        :param input_dim: 输入特征维度
        :param hidden_dim: 隐藏层维度
        :param num_layers: GRU层数 (默认2层)
        :param output_dim: 输出维度
        :param bidirectional: 是否双向 (默认True)
        :param dropout: Dropout率 (默认0.3)
        :param use_layer_norm: 是否使用层归一化 (默认True)
        """
        super(OptimizedGRU, self).__init__()
        self.num_directions = 2 if bidirectional else 1
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # GRU核心层
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )

        # 正则化组件
        self.layer_norm = nn.LayerNorm(hidden_dim * self.num_directions) if use_layer_norm else None
        self.dropout = nn.Dropout(dropout)
        
        # 输出层
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * self.num_directions, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

        # 参数初始化
        self._init_weights()

    def _init_weights(self):
        """专业参数初始化方法"""
        for name, param in self.gru.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_normal_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            if 'bias' in name:
                param.data.normal_(std=0.02)

    def forward(self, x, return_hidden=False):
        """
        前向传播
        :param x: 输入张量 (batch_size, seq_len, input_dim)
        :param return_hidden: 是否返回隐藏状态
        :return: 输出结果 (batch_size, output_dim)
        """
        batch_size = x.size(0)
        
        # 初始化隐藏状态
        h0 = torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_dim
        ).to(x.device)

        # GRU处理
        out, hidden = self.gru(x, h0)
        
        # 层归一化
        if self.layer_norm:
            out = self.layer_norm(out)
        
        # 取最后一个时间步
        out = self.dropout(out[:, -1, :])
        
        # 全连接输出
        output = self.fc(out)
        
        return (output, hidden) if return_hidden else output

    @staticmethod
    def create_optimizer(model, lr=1e-3, weight_decay=1e-4):
        """创建优化器的方法"""
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )


# 12. GRU_ekan 类
class GRU_ekan(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(GRU_ekan, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # GRU网络层
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        # 全连接层，与LSTM相同
        self.e_kan = KAN([hidden_dim, 10, output_dim])

    def forward(self, x):
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        # 前向传播GRU
        out, hn = self.gru(x, h0.detach())
        # 将GRU的最后一个时间步的输出通过全连接层
        out = self.e_kan(out[:, -1, :])
        return out

class OptimizedGRU_ekan(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(OptimizedGRU_ekan, self).__init__()  # 修改父类名称
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # GRU网络层
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        # 自定义全连接层
        self.e_kan = KAN([hidden_dim, 10, output_dim])

    def forward(self, x):
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        # 前向传播GRU
        out, hn = self.gru(x, h0.detach())
        # 将GRU的最后一个时间步的输出通过全连接层
        out = self.e_kan(out[:, -1, :])
        return out
        
class GRUwGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(GRUwGNN, self).__init__()
        self.hidden_dim = hidden_dim
        
        # GRU模块（假设输入维度正确）
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        # 图网络模块（修正后）
        self.gnn = nn.Sequential(
            nn.Linear(hidden_dim, 64),  # 确保输入维度与GRU输出一致
            nn.ReLU(),
            nn.Linear(64, 32),         # 逐步降维
            nn.ReLU(),
            nn.Linear(32, output_dim)  # 最终输出维度
        )

    def forward(self, x):
        # GRU前向传播
        gru_out, _ = self.gru(x)  # 输出形状应为 (batch, seq_len, hidden_dim)
        
        # 取最后一个时间步
        last_step = gru_out[:, -1, :]  # 形状变为 (batch, hidden_dim)
        
        # 图网络处理
        return self.gnn(last_step)     # 形状最终变为 (batch, output_dim)


class GRU_ekanwGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        # 关键修正：使用当前类名作为super参数
        super(GRU_ekanwGNN, self).__init__()  # 原错误使用了 OptimizedGRU_ekan
        
        # 网络参数初始化
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # GRU模块
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False
        )
        
        # KAN增强组件
        self.e_kan = KAN([hidden_dim, 64, 32])  # 示例维度
        
        # 图神经网络组件
        self.gnn = nn.Sequential(
            nn.Linear(32, 32),  # 输入维度需匹配KAN输出
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        # GRU处理时序特征
        gru_out, _ = self.gru(x)  # (batch, seq_len, hidden_dim)
        
        # 取最后一个时间步
        last_hidden = gru_out[:, -1, :]  # (batch, hidden_dim)
        
        # KAN增强处理
        kan_out = self.e_kan(last_hidden)
        
        # 图网络处理
        return self.gnn(kan_out)


# 13. BiLSTM 类
class BiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(BiLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # 双向LSTM网络层
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        # 注意：因为是双向，所以全连接层的输入是 hidden_dim * 2
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).requires_grad_()
        # 前向传播双向LSTM
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        # 取双向的最后一个时间步的输出
        out = self.fc(out[:, -1, :])
        return out

# 14. BiLSTM_ekan 类
class BiLSTM_ekan(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(BiLSTM_ekan, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # 双向LSTM网络层
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        # 注意：因为是双向，所以全连接层的输入是 hidden_dim * 2
        self.e_kan = KAN([hidden_dim * 2, 10, output_dim])

    def forward(self, x):
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).requires_grad_()
        # 前向传播双向LSTM
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        # 取双向的最后一个时间步的输出
        out = self.e_kan(out[:, -1, :])
        return out

# 15. TemporalConvNet 类
class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_channels, kernel_size=2, dropout=0.2, use_kan=True):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size, padding=(kernel_size-1) * dilation_size, dropout=dropout)]
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], num_outputs)  # 修改以匹配输出特征数

    def forward(self, x):
        x = x.transpose(1, 2)  # 将 batch_size, sequence_length, num_features 转换为 batch_size, num_features, sequence_length
        x = self.network(x)
        x = x[:, :, -1]  # 选择每个序列的最后一个输出
        x = self.fc(x)
        return x

# 16. TemporalConvNet_ekan 类
class TemporalConvNet_ekan(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_channels, kernel_size=2, dropout=0.2, use_kan=True):
        super(TemporalConvNet_ekan, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size, padding=(kernel_size-1) * dilation_size, dropout=dropout)]
        self.network = nn.Sequential(*layers)
        self.e_kan = KAN([num_channels[-1], 10, num_outputs])

    def forward(self, x):
        x = x.transpose(1, 2)  # 将 batch_size, sequence_length, num_features 转换为 batch_size, num_features, sequence_length
        x = self.network(x)
        x = x[:, :, -1]  # 选择每个序列的最后一个输出
        x = self.e_kan(x)
        return x

# 17. TemporalBlock 类
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

# 18. Chomp1d 类
class Chomp1d(nn.Module):
    def __init__(self, padding):
        super(Chomp1d, self).__init__()
        self.padding = padding

    def forward(self, x):
        # Chomp1d 是一个简单的自定义层，用于剪切掉因为填充(padding)导致的多余的输出，这是保证因果卷积不看到未来信息的关键。
        return x[:, :, :-self.padding]

# 19. TimeSeriesTransformer 类
class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, num_outputs, hidden_space, dropout_rate=0.1):
        super(TimeSeriesTransformer, self).__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_outputs = num_outputs
        self.hidden_space = hidden_space
        # Transformer 的 Encoder 部分
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden_space,  # 输入特征维度
            nhead=num_heads,  # 多头注意力机制的头数
            dropout=dropout_rate
        )
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)
        # 将 Encoder 的输出通过一个全连接层转换为所需的输出维度
        self.output_layer = nn.Linear(hidden_space, num_outputs)
        self.transform_layer = nn.Linear(input_dim, hidden_space)

    def forward(self, x):
        # 转换输入数据维度以符合 Transformer 的要求：(seq_len, batch_size, feature_dim)
        x = x.permute(1, 0, 2)
        x = self.transform_layer(x)
        # Transformer 编码器
        x = self.transformer_encoder(x)
        # 取最后一个时间步的输出
        x = x[-1, :, :]
        # 全连接层生成最终输出
        x = self.output_layer(x)
        return x

# 20. TimeSeriesTransformer_ekan 类
class TimeSeriesTransformer_ekan(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, num_outputs, hidden_space, dropout_rate=0.1):
        super(TimeSeriesTransformer_ekan, self).__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_outputs = num_outputs
        self.hidden_space = hidden_space
        # Transformer 的 Encoder 部分
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden_space,  # 输入特征维度
            nhead=num_heads,  # 多头注意力机制的头数
            dropout=dropout_rate
        )
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)
        # 将 Encoder 的输出通过一个全连接层转换为所需的输出维度
        self.e_kan = KAN([hidden_space, 10, num_outputs])
        self.transform_layer = nn.Linear(input_dim, hidden_space)

    def forward(self, x):
        # 转换输入数据维度以符合 Transformer 的要求：(seq_len, batch_size, feature_dim)
        x = x.permute(1, 0, 2)
        x = self.transform_layer(x)
        # Transformer 编码器
        x = self.transformer_encoder(x)
        # 取最后一个时间步的输出
        x = x[-1, :, :]
        # 全连接层生成最终输出
        x = self.e_kan(x)
        return x
class Informer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, output_dim, dropout_rate=0.1, seq_len=100, label_len=24, pred_len=24):
        super(Informer, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len

        # Encoding part (Self-Attention)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(dropout_rate)
        )

        # Informer Attention Layer (probSparse)
        self.attention = ProbSparseAttention(hidden_dim, num_heads, dropout_rate)
        
        # Decoder part
        self.decoder = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # Encode the input sequence
        x = self.encoder(x)
        
        # Pass through the sparse attention mechanism
        x = self.attention(x)
        
        # Decode the output to the required prediction shape
        x = self.decoder(x)
        
        # Ensure we only return the last time step
        return x[:, -1, :]  # Select only the last time step's output

# Define the sparse attention mechanism used in Informer
class ProbSparseAttention(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.1):
        super(ProbSparseAttention, self).__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.dropout = dropout
        
        self.q_linear = nn.Linear(dim, dim)
        self.k_linear = nn.Linear(dim, dim)
        self.v_linear = nn.Linear(dim, dim)
        
        self.attn_drop = nn.Dropout(dropout)
        self.out_linear = nn.Linear(dim, dim)
        
    def forward(self, x):
        # Calculate Q, K, V matrices
        Q = self.q_linear(x)
        K = self.k_linear(x)
        V = self.v_linear(x)
        
        # Split into multiple heads
        Q = Q.view(Q.size(0), Q.size(1), self.num_heads, self.dim // self.num_heads)
        K = K.view(K.size(0), K.size(1), self.num_heads, self.dim // self.num_heads)
        V = V.view(V.size(0), V.size(1), self.num_heads, self.dim // self.num_heads)

        # Scale factor for attention
        scale = Q.size(-1) ** 0.5
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / scale
        
        # Sparse attention with some optimization could be added here
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_drop(attn_weights)
        
        # Compute the attention output
        attn_output = torch.matmul(attn_weights, V)
        
        # Combine the heads and pass through the output layer
        attn_output = attn_output.view(attn_output.size(0), attn_output.size(1), self.dim)
        out = self.out_linear(attn_output)
        
        return out

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, BatchNorm  # 来自 torch_geometric 的 BatchNorm
from torch_geometric.utils import dense_to_sparse

class TimeGNN(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        seq_len,
        batch_size,
        aggregate="last",
        keep_self_loops=False,
        enforce_consecutive=False,
        block_size=3
    ):
        super().__init__()
        # 初始化参数
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.aggregate = aggregate
        self.keep_self_loops = keep_self_loops
        self.enforce_consecutive = enforce_consecutive
        self.block_size = block_size
        
        # 模型其余初始化代码...
        
        # 动态序列跟踪
        self.register_buffer('current_seq_len', torch.tensor(0))
        
        # 特征提取层
        self.conv_branch1 = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, 1, padding="same"),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding="same", dilation=3),
            nn.ReLU()
        )
        self.conv_branch2 = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, 1, padding="same"),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 5, padding="same", dilation=5),
            nn.ReLU()
        )
        self.conv_branch3 = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, 1, padding="same"),
            nn.ReLU()
        )
        self.feature_fusion = nn.Linear(hidden_dim*3, hidden_dim)
        
        # 边学习组件（延迟初始化）
        self.register_buffer('rec_idx', None)
        self.register_buffer('send_idx', None)
        self.edge_fc1 = nn.Linear(hidden_dim*2, hidden_dim)
        self.edge_fc2 = nn.Linear(hidden_dim, 2)
        
        # 动态掩码（延迟初始化）
        self.register_buffer('tri_mask', None)
        self.register_buffer('diagonal_mask', None)
        
        # GNN模块
        self.gnns = nn.ModuleList([
            SAGEConv(hidden_dim, hidden_dim, normalize=False)
            for _ in range(block_size)
        ])
        self.bns = nn.ModuleList([
            BatchNorm(hidden_dim) for _ in range(block_size)
        ])
        self.gnn_weight = nn.Linear(block_size, 1)
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.BatchNorm1d(hidden_dim//2),  # 使用 PyTorch 原生的 BatchNorm1d
            nn.ReLU(),
            nn.Linear(hidden_dim//2, output_dim)
        )
        
        # 参数初始化
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _init_dynamic_components(self, seq_len, device):
        """动态初始化序列相关组件"""
        if self.current_seq_len == seq_len and self.rec_idx is not None:
            return
        
        # 更新当前序列长度
        self.current_seq_len = torch.tensor(seq_len, device=device)
        
        # 生成接收/发送索引
        indices = torch.arange(seq_len, device=device)
        self.rec_idx = F.one_hot(indices.repeat(seq_len), seq_len).float()
        self.send_idx = F.one_hot(indices.repeat_interleave(seq_len), seq_len).float()
        
        # 生成动态掩码
        self.tri_mask = ~torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=-1)
        self.diagonal_mask = torch.eye(seq_len, dtype=torch.bool, device=device)
        
    def forward(self, data, return_graphs=False):
        # 输入维度处理 [batch_size, seq_len, features]
        if data.dim() == 2:
            data = data.unsqueeze(1)
        batch_size, seq_len, features = data.size()
        data = data.permute(0, 2, 1)  # [batch, features, seq_len]
        
        # 动态初始化组件
        self._init_dynamic_components(seq_len, data.device)
        
        # 特征提取
        x1 = self.conv_branch1(data)  # [batch, hid, seq]
        x2 = self.conv_branch2(data)
        x3 = self.conv_branch3(data)
        
        # 特征融合
        x = torch.cat([x1, x2, x3], dim=1).permute(0, 2, 1)  # [batch, seq, 3*hid]
        x = self.feature_fusion(x)  # [batch, seq, hid]
        x = F.relu(x)
        
        # 边学习
        receivers = torch.bmm(
            self.rec_idx.repeat(batch_size,1,1),  # [batch, seq^2, seq]
            x  # [batch, seq, hid]
        )  # [batch, seq^2, hid]
        
        senders = torch.matmul(
            self.send_idx,  # [seq^2, seq]
            x  # [batch, seq, hid]
        )  # [batch, seq^2, hid]
        
        edge_features = torch.cat([senders, receivers], dim=-1)  # [batch, seq^2, 2*hid]
        edges = F.relu(self.edge_fc1(edge_features))  # [batch, seq^2, hid]
        edges = self.edge_fc2(edges)  # [batch, seq^2, 2]
        
        # 邻接矩阵生成
        adj = F.gumbel_softmax(edges, tau=0.5, hard=True)[..., 0]  # [batch, seq^2]
        adj = adj.view(batch_size, seq_len, seq_len)
        
        # 应用掩码
        adj = adj.masked_fill(self.tri_mask, 0)
        if not self.keep_self_loops:
            adj = adj.masked_fill(self.diagonal_mask, 0)
        
        # GNN处理
        x = x.view(-1, self.hidden_dim)  # [batch*seq, hid]
        edge_index, _ = dense_to_sparse(adj)  # [2, num_edges]
        
        gnn_outputs = []
        for gnn, bn in zip(self.gnns, self.bns):
            x = gnn(x, edge_index)
            x = bn(x)
            gnn_outputs.append(F.relu(x))
        
        # 多尺度融合
        x = torch.stack(gnn_outputs, dim=-1)  # [batch*seq, hid, block]
        x = self.gnn_weight(x).squeeze(-1)  # [batch*seq, hid]
        x = x.view(batch_size, seq_len, self.hidden_dim)  # [batch, seq, hid]
        
        # 聚合输出
        if self.aggregate == "mean":
            x = x.mean(dim=1)
        elif self.aggregate == "last":
            x = x[:, -1]
        else:
            raise ValueError(f"Unsupported aggregation: {self.aggregate}")
        
        output = self.output_layer(x)
        return (output, adj) if return_graphs else output

   