from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F


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


#############
# New Impls #
#############
def init_rate_half(rate):
    if rate is not None:
        rate.data.fill_(0.5)


def init_rate_0(tensor):
    if tensor is not None:
        tensor.data.fill_(0.)
    return tensor


def position(H, W, is_cuda=True):
    if is_cuda:
        loc_w = torch.linspace(-1.0, 1.0, W).cuda().unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).cuda().unsqueeze(1).repeat(1, W)
    else:
        loc_w = torch.linspace(-1.0, 1.0, W).unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).unsqueeze(1).repeat(1, W)
    loc = torch.cat([loc_w.unsqueeze(0), loc_h.unsqueeze(0)], 0).unsqueeze(0)
    return loc


def position1d(length):
    loc = torch.linspace(-1.0, 1.0, length).unsqueeze(0)
    loc = loc.unsqueeze(0)
    return loc


def stride(x, stride):
    b, c, w = x.shape
    return x[:, :, ::stride]


class ACmix(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_att=7, head=4, kernel_conv=3, stride=1, dilation=1):
        super(ACmix, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.head = head
        self.kernel_att = kernel_att
        self.kernel_conv = kernel_conv
        self.stride = stride
        self.dilation = dilation
        self.rate1 = torch.nn.Parameter(torch.Tensor(1))
        self.rate2 = torch.nn.Parameter(torch.Tensor(1))
        self.head_dim = self.out_planes // self.head

        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv3 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv_p = nn.Conv2d(2, self.head_dim, kernel_size=1)

        self.padding_att = (self.dilation * (self.kernel_att - 1) + 1) // 2
        self.pad_att = torch.nn.ReflectionPad2d(self.padding_att)
        self.unfold = nn.Unfold(kernel_size=self.kernel_att, padding=0, stride=self.stride)
        self.softmax = torch.nn.Softmax(dim=1)

        self.fc = nn.Conv2d(3 * self.head, self.kernel_conv * self.kernel_conv, kernel_size=1, bias=False)
        self.dep_conv = nn.Conv2d(self.kernel_conv * self.kernel_conv * self.head_dim, out_planes,
                                  kernel_size=self.kernel_conv, bias=True, groups=self.head_dim, padding=1,
                                  stride=stride)

        self.reset_parameters()

    def reset_parameters(self):
        init_rate_half(self.rate1)
        init_rate_half(self.rate2)
        kernel = torch.zeros(self.kernel_conv * self.kernel_conv, self.kernel_conv, self.kernel_conv)
        for i in range(self.kernel_conv * self.kernel_conv):
            kernel[i, i // self.kernel_conv, i % self.kernel_conv] = 1.
        kernel = kernel.squeeze(0).repeat(self.out_planes, 1, 1, 1)
        self.dep_conv.weight = nn.Parameter(data=kernel, requires_grad=True)
        self.dep_conv.bias = init_rate_0(self.dep_conv.bias)

    def forward(self, x):
        q, k, v = self.conv1(x), self.conv2(x), self.conv3(x)
        scaling = float(self.head_dim) ** -0.5
        b, c, h, w = q.shape
        h_out, w_out = h // self.stride, w // self.stride

        # ### att
        # ## positional encoding
        pe = self.conv_p(position(h, w))

        q_att = q.view(b * self.head, self.head_dim, h, w) * scaling
        k_att = k.view(b * self.head, self.head_dim, h, w)
        v_att = v.view(b * self.head, self.head_dim, h, w)

        if self.stride > 1:
            q_att = stride(q_att, self.stride)
            q_pe = stride(pe, self.stride)
        else:
            q_pe = pe

        unfold_k = self.unfold(self.pad_att(k_att)).view(b * self.head, self.head_dim,
                                                         self.kernel_att * self.kernel_att, h_out,
                                                         w_out)  # b*head, head_dim, k_att^2, h_out, w_out
        unfold_rpe = self.unfold(self.pad_att(pe)).view(1, self.head_dim, self.kernel_att * self.kernel_att, h_out,
                                                        w_out)  # 1, head_dim, k_att^2, h_out, w_out

        # (b*head, head_dim, 1, h_out, w_out) * (b*head, head_dim, k_att^2, h_out, w_out)
        #   -> (b*head, k_att^2, h_out, w_out)
        att = (q_att.unsqueeze(2) * (unfold_k + q_pe.unsqueeze(2) - unfold_rpe)).sum(1)
        att = self.softmax(att)

        out_att = self.unfold(self.pad_att(v_att)).view(b * self.head, self.head_dim, self.kernel_att * self.kernel_att,
                                                        h_out, w_out)
        out_att = (att.unsqueeze(1) * out_att).sum(2).view(b, self.out_planes, h_out, w_out)

        ## conv
        f_all = self.fc(torch.cat(
            [q.view(b, self.head, self.head_dim, h * w), k.view(b, self.head, self.head_dim, h * w),
             v.view(b, self.head, self.head_dim, h * w)], 1))
        f_conv = f_all.permute(0, 2, 1, 3).reshape(x.shape[0], -1, x.shape[-2], x.shape[-1])

        out_conv = self.dep_conv(f_conv)

        return self.rate1 * out_att + self.rate2 * out_conv


def reflection_pad1d(input, padding):
    """
    Apply reflection padding to a 1D tensor.
    Args:
        input (torch.Tensor): Input tensor of shape (B, C, W)
        padding (int): Amount of padding on both sides
    Returns:
        torch.Tensor: Padded tensor
    """
    if padding == 0:
        return input
    # Padding left
    left_pad = input[..., 1:padding + 1].flip(dims=[-1])
    # Padding right
    right_pad = input[..., -padding - 1:-1].flip(dims=[-1])
    # Concatenate along the last dimension
    return torch.cat([left_pad, input, right_pad], dim=-1)


class ACmix1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_att=7, num_heads=4, conv_kernel_size=3, stride=1, dilation=1):
        super(ACmix1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.kernel_att = kernel_att
        self.conv_kernel_size = conv_kernel_size
        self.stride = stride
        self.dilation = dilation
        self.rate1 = torch.nn.Parameter(torch.Tensor(1))
        self.rate2 = torch.nn.Parameter(torch.Tensor(1))
        self.head_dim = self.out_channels // self.num_heads

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.conv_p = nn.Conv1d(1, self.head_dim, kernel_size=1)

        self.padding_att = (self.dilation * (self.kernel_att - 1) + 1) // 2
        self.softmax = nn.Softmax(dim=1)

        self.fc = nn.Conv2d(3 * self.num_heads, self.conv_kernel_size, kernel_size=1, bias=False)
        self.dep_conv = nn.Conv1d(self.conv_kernel_size * self.head_dim, out_channels,
                                  kernel_size=self.conv_kernel_size, bias=True, groups=self.head_dim, padding=0,
                                  stride=stride)

        self.reset_parameters()

    def reset_parameters(self):
        init_rate_half(self.rate1)
        init_rate_half(self.rate2)
        kernel = torch.zeros(self.conv_kernel_size, self.conv_kernel_size)
        for i in range(self.conv_kernel_size):
            kernel[i, i] = 1.
        kernel = kernel.unsqueeze(0).repeat(self.out_channels, 1, 1)
        self.dep_conv.weight = nn.Parameter(data=kernel, requires_grad=True)
        self.dep_conv.bias = init_rate_0(self.dep_conv.bias)

    def forward(self, x):
        q, k, v = self.conv1(x), self.conv2(x), self.conv3(x)
        scaling = float(self.head_dim) ** -0.5
        b, c, w = q.shape
        w_out = w // self.stride

        # Positional encoding: head_dim,
        pe = self.conv_p(position1d(w).to(x.device))

        q_att = q.view(b * self.num_heads, self.head_dim, w) * scaling
        k_att = k.view(b * self.num_heads, self.head_dim, w)
        v_att = v.view(b * self.num_heads, self.head_dim, w)

        if self.stride > 1:
            q_att = stride(q_att, self.stride)
            q_pe = stride(pe, self.stride)
        else:
            q_pe = pe

        # Custom unfold operation for 1D
        k_att_padded = reflection_pad1d(k_att, self.padding_att)
        unfold_k = k_att_padded.unfold(dimension=-1, size=self.kernel_att, step=self.stride)
        unfold_k = unfold_k.permute(0, 1, 3, 2).contiguous()  # b*head, head_dim, k_att, w_out

        pe_padded = reflection_pad1d(pe, self.padding_att)
        unfold_rpe = pe_padded.unfold(dimension=-1, size=self.kernel_att, step=self.stride)
        unfold_rpe = unfold_rpe.permute(0, 1, 3, 2).contiguous()

        q_pe = q_pe.unsqueeze(2)

        # Attention calculation
        q_att = q_att.unsqueeze(2)  # (b * head, head_dim, 1, w_out)
        att = (q_att * (unfold_k + q_pe - unfold_rpe)).sum(1)  # (b * head, kernel_att, w_out)
        att = self.softmax(att)

        v_att_padded = reflection_pad1d(v_att, self.padding_att)
        unfold_v = v_att_padded.unfold(dimension=-1, size=self.kernel_att, step=self.stride)
        unfold_v = unfold_v.permute(0, 1, 3, 2).contiguous()
        out_att = (att.unsqueeze(1) * unfold_v).sum(2).view(b, self.out_channels, w_out)

        # Convolution
        f_all = self.fc(torch.cat(
            [q.view(b, self.num_heads, self.head_dim, w), k.view(b, self.num_heads, self.head_dim, w),
             v.view(b, self.num_heads, self.head_dim, w)], 1))
        f_conv = f_all.permute(0, 2, 1, 3).reshape(x.shape[0], -1, x.shape[-1])

        out_conv = self.dep_conv(f_conv)

        return self.rate1.to(x.device) * out_att + self.rate2.to(x.device) * out_conv


class ACResnet1d(nn.Module):
    """
    1D ResNet with attention and convolutional blocks.
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

        # Prefiltering
        prefilt_padding = prefilt_kernel_size // 2
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=n_in,
                      out_channels=n_ch[0],
                      kernel_size=prefilt_kernel_size,
                      padding=prefilt_padding,
                      stride=1),
            activation_layer(),
            ACmix1d(in_channels=n_ch[0],
                    out_channels=n_ch[0],
                    kernel_att=3,
                    num_heads=4,
                    conv_kernel_size=1),
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


if __name__ == '__main__':
    x = torch.randn(12, 1, 264)
    model = ACResnet1d(a_lrelu=0.3, activation_fn='leaky', n_bins_in=264,
                       n_chan_input=1, n_chan_layers=(40, 30, 30, 10, 3),
                       n_prefilt_layers=2, output_dim=384, p_dropout=0.2,
                       prefilt_kernel_size=15, residual=True)
    model.forward(x)
