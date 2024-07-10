from functools import partial

import torch
import math
import torch.nn.functional as F
import torch.nn as nn


class ToeplitzLinear(nn.Conv1d):
    def __init__(self, in_features, out_features):
        super(ToeplitzLinear, self).__init__(
            in_channels=1,
            out_channels=1,
            kernel_size=in_features + out_features - 1,
            padding=out_features - 1,
            bias=False
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return super(ToeplitzLinear, self).forward(input.unsqueeze(-2)).squeeze(-2)


class RelativePosition(nn.Module):

    def __init__(self, num_units, max_relative_position):
        super().__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position
        self.embeddings_table = nn.Parameter(torch.Tensor(max_relative_position * 2 + 1, num_units))
        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, length_q, length_k):
        range_vec_q = torch.arange(length_q)
        range_vec_k = torch.arange(length_k)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
        final_mat = distance_mat_clipped + self.max_relative_position
        final_mat = torch.LongTensor(final_mat)
        embeddings = self.embeddings_table[final_mat]

        return embeddings


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout):
        super().__init__()

        assert hid_dim % n_heads == 0

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        self.max_relative_position = 2

        self.relative_position_k = RelativePosition(self.head_dim, self.max_relative_position)
        # self.relative_position_v = RelativePosition(self.head_dim, self.max_relative_position)

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)

        self.fc_o = nn.Linear(hid_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]))

    def forward(self, query, key, value, mask=None):
        # query = [batch size, query len, hid dim]
        # key = [batch size, key len, hid dim]
        # value = [batch size, value len, hid dim]
        batch_size = query.shape[0]
        len_k = key.shape[1]
        len_q = query.shape[1]
        len_v = value.shape[1]

        query = self.fc_q(query)
        key = self.fc_k(key)
        value = self.fc_v(value)

        r_q1 = query.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        r_k1 = key.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        attn1 = torch.matmul(r_q1, r_k1.permute(0, 1, 3, 2)).to(query.device)

        r_q2 = query.permute(1, 0, 2).contiguous().view(len_q, batch_size * self.n_heads, self.head_dim)
        r_k2 = self.relative_position_k(len_q, len_k)
        attn2 = torch.matmul(r_q2, r_k2.transpose(1, 2)).transpose(0, 1)
        attn2 = attn2.contiguous().view(batch_size, self.n_heads, len_q, len_k).to(query.device)
        attn = (attn1 + attn2) / self.scale.to(query.device)

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e10)

        attn = self.dropout(torch.softmax(attn, dim=-1))

        # attn = [batch size, n heads, query len, key len]
        r_v1 = value.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        weight1 = torch.matmul(attn, r_v1)
        # r_v2 = self.relative_position_v(len_q, len_v)
        # weight2 = attn.permute(2, 0, 1, 3).contiguous().view(len_q, batch_size * self.n_heads, len_k)
        # weight2 = torch.matmul(weight2, r_v2)
        # weight2 = weight2.transpose(0, 1).contiguous().view(batch_size, self.n_heads, len_q, self.head_dim)

        x = weight1  # + weight2

        # x = [batch size, n heads, query len, head dim]

        x = x.permute(0, 2, 1, 3).contiguous()

        # x = [batch size, query len, n heads, head dim]

        x = x.view(batch_size, -1, self.hid_dim)

        # x = [batch size, query len, hid dim]

        x = self.fc_o(x)

        # x = [batch size, query len, hid dim]

        return x


class RelativeGlobalAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_len=1024, dropout=0.1):
        super().__init__()
        d_head, remainder = divmod(d_model, num_heads)
        if remainder:
            raise ValueError(
                "incompatible `d_model` and `num_heads`"
            )
        self.max_len = max_len
        self.d_model = d_model
        self.num_heads = num_heads
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.query = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.Er = nn.Parameter(torch.randn(max_len, d_head))
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(max_len, max_len))
            .unsqueeze(0).unsqueeze(0)
        )
        # self.mask.shape = (1, 1, max_len, max_len)

    def forward(self, x):
        # x.shape == (batch_size, seq_len, d_model)
        batch_size, seq_len, _ = x.shape

        if seq_len > self.max_len:
            raise ValueError(
                "sequence length exceeds model capacity"
            )

        k_t = self.key(x).reshape(batch_size, seq_len, self.num_heads, -1).permute(0, 2, 3, 1)
        # k_t.shape = (batch_size, num_heads, d_head, seq_len)
        v = self.value(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        q = self.query(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        # shape = (batch_size, num_heads, seq_len, d_head)

        start = self.max_len - seq_len
        Er_t = self.Er[start:, :].transpose(0, 1)
        # Er_t.shape = (d_head, seq_len)
        QEr = torch.matmul(q, Er_t)
        # QEr.shape = (batch_size, num_heads, seq_len, seq_len)
        Srel = self.skew(QEr)
        # Srel.shape = (batch_size, num_heads, seq_len, seq_len)

        QK_t = torch.matmul(q, k_t)
        # QK_t.shape = (batch_size, num_heads, seq_len, seq_len)
        attn = (QK_t + Srel) / math.sqrt(q.size(-1))
        mask = self.mask[:, :, :seq_len, :seq_len]
        # mask.shape = (1, 1, seq_len, seq_len)
        attn = attn.masked_fill(mask == 0, float("-inf"))
        # attn.shape = (batch_size, num_heads, seq_len, seq_len)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        # out.shape = (batch_size, num_heads, seq_len, d_head)
        out = out.transpose(1, 2)
        # out.shape == (batch_size, seq_len, num_heads, d_head)
        out = out.reshape(batch_size, seq_len, -1)
        # out.shape == (batch_size, seq_len, d_model)
        return self.dropout(out)

    def skew(self, QEr):
        # QEr.shape = (batch_size, num_heads, seq_len, seq_len)
        padded = F.pad(QEr, (1, 0))
        # padded.shape = (batch_size, num_heads, seq_len, 1 + seq_len)
        batch_size, num_heads, num_rows, num_cols = padded.shape
        reshaped = padded.reshape(batch_size, num_heads, num_cols, num_rows)
        # reshaped.size = (batch_size, num_heads, 1 + seq_len, seq_len)
        Srel = reshaped[:, :, 1:, :]
        # Srel.shape = (batch_size, num_heads, seq_len, seq_len)
        return Srel


class PosCNN(nn.Module):
    def __init__(self, in_chans, embed_dim=768, s=1):
        super(PosCNN, self).__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size=(3, 1), stride=(s, 1), padding=(1, 0), bias=True,
                      groups=embed_dim),
            nn.ReLU(inplace=True)
        )
        self.s = s

    def forward(self, x, H, W):
        B, N, C = x.shape
        feat_token = x
        cnn_feat = feat_token.view(B, N, C, 1)  # reshape to [batch_size, sequence_length, channels, 1]
        cnn_feat = self.proj(cnn_feat) + cnn_feat if self.s == 1 else self.proj(cnn_feat)
        x = cnn_feat.flatten(2).transpose(1, 2)
        return x


class PosCNN1D(nn.Module):
    def __init__(self, in_chans, embed_dim=768, s=1):
        super(PosCNN1D, self).__init__()
        self.proj = nn.Conv1d(in_chans, embed_dim, 3, stride=s, padding=1, bias=True, groups=embed_dim)
        self.s = s

    def forward(self, x):
        B, C, L = x.shape
        cnn_feat = x
        if self.s == 1:
            x = self.proj(cnn_feat) + cnn_feat
        else:
            x = self.proj(cnn_feat)
        return x

class  PEG(nn.Module):
      def __init__(self, in_channels, embed_dim, kernel_size=3, stride=1):
        super(PEG, self).__init__()
        self.proj = nn.Conv1d(in_channels, embed_dim, kernel_size, stride, padding=kernel_size//2, groups=in_channels, bias=True)

      def forward(self, x):
        return self.proj(x) + x


class Resnet1d(nn.Module):
    """
    Basic CNN similar to the one in Johannes Zeitler's report,
    but for longer HCQT input (always stride 1 in time)
    Still with 75 (-1) context frames, i.e. 37 frames padded to each side
    The number of input channels, channels in the hidden layers, and output
    dimensions (e.g. for pitch output) can be parameterized.
    Layer normalization is only performed over frequency and channel dimensions,
    not over time (in order to work with variable length input).
    Outputs one channel with sigmoid activation.

    Args (Defaults: BasicCNN by Johannes Zeitler but with 6 input channels):
        n_chan_input:     Number of input channels (harmonics in HCQT)
        n_chan_layers:    Number of channels in the hidden layers (list)
        n_prefilt_layers: Number of repetitions of the prefiltering layer
        residual:         If True, use residual connections for prefiltering (default: False)
        n_bins_in:        Number of input bins (12 * number of octaves)
        n_bins_out:       Number of output bins (12 for pitch class, 72 for pitch, num_octaves * 12)
        a_lrelu:          alpha parameter (slope) of LeakyReLU activation function
        p_dropout:        Dropout probability
    """

    def __init__(self,
                 n_chan_input=1,
                 n_chan_layers=(20, 20, 10, 1),
                 n_prefilt_layers=1,
                 prefilt_kernel_size=15,
                 residual=False,
                 n_bins_in=216,
                 output_dim=128,
                 activation_fn: str = "leaky",
                 a_lrelu=0.3,
                 p_dropout=0.2):
        super(Resnet1d, self).__init__()

        self.hparams = dict(n_chan_input=n_chan_input,
                            n_chan_layers=n_chan_layers,
                            n_prefilt_layers=n_prefilt_layers,
                            prefilt_kernel_size=prefilt_kernel_size,
                            residual=residual,
                            n_bins_in=n_bins_in,
                            output_dim=output_dim,
                            activation_fn=activation_fn,
                            a_lrelu=a_lrelu,
                            p_dropout=p_dropout)

        if activation_fn == "relu":
            activation_layer = nn.ReLU
        elif activation_fn == "silu":
            activation_layer = nn.SiLU
        elif activation_fn == "leaky":
            activation_layer = partial(nn.LeakyReLU, negative_slope=a_lrelu)
        else:
            raise ValueError

        n_in = n_chan_input
        n_ch = n_chan_layers
        if len(n_ch) < 5:
            n_ch.append(1)

        # Layer normalization over frequency and channels (harmonics of HCQT)
        self.layernorm = nn.LayerNorm(normalized_shape=[n_in, n_bins_in])

        # Prefiltering
        prefilt_padding = prefilt_kernel_size // 2
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=n_in,
                      out_channels=n_ch[0],
                      kernel_size=prefilt_kernel_size,
                      padding=prefilt_padding,
                      stride=1),
            activation_layer(),
            nn.Dropout(p=p_dropout)
        )
        self.n_prefilt_layers = n_prefilt_layers
        self.prefilt_layers = nn.ModuleList(*[
            nn.Sequential(
                nn.Conv1d(in_channels=n_ch[0],
                          out_channels=n_ch[0],
                          kernel_size=prefilt_kernel_size,
                          padding=prefilt_padding,
                          stride=1),
                activation_layer(),
                nn.Dropout(p=p_dropout)
            )
            for _ in range(n_prefilt_layers - 1)
        ])
        self.residual = residual

        conv_layers = []
        for i in range(len(n_chan_layers) - 1):
            conv_layers.extend([
                nn.Conv1d(in_channels=n_ch[i],
                          out_channels=n_ch[i + 1],
                          kernel_size=1,
                          padding=0,
                          stride=1),
                activation_layer(),
                nn.Dropout(p=p_dropout)
            ])
        self.conv_layers = nn.Sequential(*conv_layers)

        self.flatten = nn.Flatten(start_dim=1)
        self.fc = ToeplitzLinear(n_bins_in * n_ch[-1], output_dim)

        self.final_norm = nn.Softmax(dim=-1)

    def forward(self, x):
        r"""

        Args:
            x (torch.Tensor): shape (batch, channels, freq_bins)
        """
        x = self.layernorm(x)

        x = self.conv1(x)
        for p in range(0, self.n_prefilt_layers - 1):
            prefilt_layer = self.prefilt_layers[p]
            if self.residual:
                x_new = prefilt_layer(x)
                x = x_new + x
            else:
                x = prefilt_layer(x)

        x = self.conv_layers(x)
        x = self.flatten(x)

        y_pred = self.fc(x)

        return self.final_norm(y_pred)


class Resnet1dWtAttention(nn.Module):
    """
    Basic CNN similar to the one in Johannes Zeitler's report,
    but for longer HCQT input (always stride 1 in time)
    Still with 75 (-1) context frames, i.e. 37 frames padded to each side
    The number of input channels, channels in the hidden layers, and output
    dimensions (e.g. for pitch output) can be parameterized.
    Layer normalization is only performed over frequency and channel dimensions,
    not over time (in order to work with variable length input).
    Outputs one channel with sigmoid activation.

    Args (Defaults: BasicCNN by Johannes Zeitler but with 6 input channels):
        n_chan_input:     Number of input channels (harmonics in HCQT)
        n_chan_layers:    Number of channels in the hidden layers (list)
        n_prefilt_layers: Number of repetitions of the prefiltering layer
        residual:         If True, use residual connections for prefiltering (default: False)
        n_bins_in:        Number of input bins (12 * number of octaves)
        n_bins_out:       Number of output bins (12 for pitch class, 72 for pitch, num_octaves * 12)
        a_lrelu:          alpha parameter (slope) of LeakyReLU activation function
        p_dropout:        Dropout probability
    """

    def __init__(self,
                 n_chan_input=1,
                 n_chan_layers=(20, 20, 10, 1),
                 n_prefilt_layers=1,
                 prefilt_kernel_size=15,
                 residual=False,
                 n_bins_in=216,
                 output_dim=128,
                 activation_fn: str = "leaky",
                 a_lrelu=0.3,
                 p_dropout=0.2,
                 embed_dim=16,
                 num_heads=4):
        super().__init__()

        self.hparams = dict(n_chan_input=n_chan_input,
                            n_chan_layers=n_chan_layers,
                            n_prefilt_layers=n_prefilt_layers,
                            prefilt_kernel_size=prefilt_kernel_size,
                            residual=residual,
                            n_bins_in=n_bins_in,
                            output_dim=output_dim,
                            activation_fn=activation_fn,
                            a_lrelu=a_lrelu,
                            p_dropout=p_dropout)

        if activation_fn == "relu":
            activation_layer = nn.ReLU
        elif activation_fn == "silu":
            activation_layer = nn.SiLU
        elif activation_fn == "leaky":
            activation_layer = partial(nn.LeakyReLU, negative_slope=a_lrelu)
        else:
            raise ValueError

        n_in = n_chan_input
        n_ch = n_chan_layers
        if len(n_ch) < 5:
            n_ch.append(1)

        # Layer normalization over frequency and channels (harmonics of HCQT)
        self.layernorm = nn.LayerNorm(normalized_shape=[n_in, n_bins_in])
        self.n_bins_in = n_bins_in

        # Prefiltering
        prefilt_padding = prefilt_kernel_size // 2
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=n_in,
                      out_channels=n_ch[0],
                      kernel_size=prefilt_kernel_size,
                      padding=prefilt_padding,
                      stride=1),
            activation_layer(),
            nn.Dropout(p=p_dropout)
        )
        self.n_prefilt_layers = n_prefilt_layers
        self.prefilt_layers = nn.ModuleList(*[
            nn.Sequential(
                nn.Conv1d(in_channels=n_ch[0],
                          out_channels=n_ch[0],
                          kernel_size=prefilt_kernel_size,
                          padding=prefilt_padding,
                          stride=1),
                activation_layer(),
                nn.Dropout(p=p_dropout)
            )
            for _ in range(n_prefilt_layers - 1)
        ])
        self.residual = residual

        conv_layers = []
        for i in range(len(n_chan_layers) - 1):
            conv_layers.extend([
                nn.Conv1d(in_channels=n_ch[i],
                          out_channels=n_ch[i + 1],
                          kernel_size=1,
                          padding=0,
                          stride=1),
                activation_layer(),
                nn.Dropout(p=p_dropout)
            ])
        self.conv_layers = nn.Sequential(*conv_layers)

        self.proj = nn.Linear(n_ch[-1] * n_bins_in, embed_dim * n_bins_in)
        # self.attention = MultiHeadAttentionLayer(hid_dim=embed_dim, n_heads=num_heads, dropout=p_dropout)
        self.attention = RelativeGlobalAttention(d_model=embed_dim, num_heads=num_heads, dropout=p_dropout)
        self.embed_dim = embed_dim

        self.flatten = nn.Flatten(start_dim=1)
        self.fc = ToeplitzLinear(n_bins_in * embed_dim, output_dim)
        # self.fc = ToeplitzLinear(n_bins_in * n_ch[-1], output_dim)
        self.final_norm = nn.Softmax(dim=-1)

    def forward(self, x):
        r"""

        Args:
            x (torch.Tensor): shape (batch, channels, freq_bins)
        """
        x = self.layernorm(x)

        x = self.conv1(x)
        for p in range(0, self.n_prefilt_layers - 1):
            prefilt_layer = self.prefilt_layers[p]
            if self.residual:
                x_new = prefilt_layer(x)
                x = x_new + x
            else:
                x = prefilt_layer(x)

        x = self.conv_layers(x)

        # Apply MultiheadAttention:
        x = self.flatten(x)
        x = self.proj(x)
        x = x.view(x.size(0), self.embed_dim, self.n_bins_in)
        x = x.permute(0, 2, 1)  # [batch size, seq len, hid dim]
        x = self.attention(x)

        x = self.flatten(x)

        y_pred = self.fc(x)

        return self.final_norm(y_pred)
