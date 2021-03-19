import math
import numpy as np

import paddle
import paddle.nn as nn
import paddle.nn.functional as F


def swish(input):
    return input * F.sigmoid(input)


def _calculate_fan_in_and_fan_out(tensor):
    dimensions = tensor.dim()
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")

    if dimensions == 2:  # Linear # The weight axes are different from pytorch's
        fan_in = tensor.shape[0]
        fan_out = tensor.shape[1]
    else:
        num_input_fmaps = tensor.shape[1]
        num_output_fmaps = tensor.shape[0]
        receptive_field_size = 1
        if tensor.dim() > 2:
            receptive_field_size = tensor[0][0].reshape((-1, )).shape[0]
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


@paddle.no_grad()
def variance_scaling_init_(tensor, scale=1, mode="fan_avg", distribution="uniform"):
    stop_gradient = tensor.stop_gradient
    tensor.stop_gradient = True
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)

    if mode == "fan_in":
        scale /= fan_in

    elif mode == "fan_out":
        scale /= fan_out

    else:
        scale /= (fan_in + fan_out) / 2

    if distribution == "normal":
        std = math.sqrt(scale)

        tensor[:] = paddle.normal(0, std, shape=tensor.shape).astype(tensor.dtype)
        tensor.stop_gradient = stop_gradient
        return tensor

    else:
        bound = math.sqrt(3 * scale)

        tensor[:] = paddle.uniform(tensor.shape, dtype=tensor.dtype, min=-bound, max=bound)
        tensor.stop_gradient = stop_gradient
        return tensor


def conv2d(
    in_channel,
    out_channel,
    kernel_size,
    stride=1,
    padding=0,
    bias=True,
    scale=1,
    mode="fan_avg",
):
    conv = nn.Conv2D(
        in_channel, out_channel, kernel_size, stride=stride, padding=padding, bias_attr=bias
    )

    variance_scaling_init_(conv.weight, scale, mode=mode)

    if bias:
        nn.initializer.Constant(0.0)(conv.bias)

    return conv


def linear(in_channel, out_channel, scale=1, mode="fan_avg"):
    lin = nn.Linear(in_channel, out_channel)

    variance_scaling_init_(lin.weight, scale, mode=mode)
    nn.initializer.Constant(0.0)(lin.bias)

    return lin


class Swish(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return swish(input)


class Upsample(nn.Sequential):
    def __init__(self, channel):
        layers = [
            nn.Upsample(scale_factor=2, mode="nearest"),
            conv2d(channel, channel, 3, padding=1),
        ]

        super().__init__(*layers)


class Downsample(nn.Sequential):
    def __init__(self, channel):
        layers = [conv2d(channel, channel, 3, stride=2, padding=1)]

        super().__init__(*layers)


class ResBlock(nn.Layer):
    def __init__(
        self, in_channel, out_channel, time_dim, use_affine_time=False, dropout=0
    ):
        super().__init__()

        self.use_affine_time = use_affine_time
        time_out_dim = out_channel
        time_scale = 1
        norm_affine = None

        if self.use_affine_time:
            time_out_dim *= 2
            time_scale = 1e-10
            norm_affine = False

        self.norm1 = nn.GroupNorm(32, in_channel)
        self.activation1 = Swish()
        self.conv1 = conv2d(in_channel, out_channel, 3, padding=1)

        self.time = nn.Sequential(
            Swish(), linear(time_dim, time_out_dim, scale=time_scale)
        )

        # A bug if we set `weight_attr` and `bias_attr` to False.
        # Delete the weights to fix it as a temporary solution.
        # self.norm2 = nn.GroupNorm(32, out_channel, weight_attr=norm_affine, bias_attr=norm_affine)
        self.norm2 = nn.GroupNorm(32, out_channel)
        if self.use_affine_time:
            del self.norm2.weight
            del self.norm2.bias
            self.norm2.weight = None
            self.norm2.bias = None

        self.activation2 = Swish()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = conv2d(out_channel, out_channel, 3, padding=1, scale=1e-10)

        if in_channel != out_channel:
            self.skip = conv2d(in_channel, out_channel, 1)

        else:
            self.skip = None

    def forward(self, input, time):
        batch = input.shape[0]

        out = self.conv1(self.activation1(self.norm1(input)))

        if self.use_affine_time:
            gamma, beta = self.time(time).reshape((batch, -1, 1, 1)).split(2, 1)
            out = (1 + gamma) * self.norm2(out) + beta
            
        else:
            out = out + self.time(time).reshape((batch, -1, 1, 1))
            out = self.norm2(out)

        out = self.conv2(self.dropout(self.activation2(out)))

        if self.skip is not None:
            input = self.skip(input)

        return out + input


class VanillaSelfAttention(nn.Layer):
    def __init__(self, in_channel, n_head=1):
        super().__init__()

        self.n_head = n_head

        self.norm = nn.GroupNorm(32, in_channel)
        self.qkv = conv2d(in_channel, in_channel * 3, 1)
        self.out = conv2d(in_channel, in_channel, 1, scale=1e-10)
        self.key_dim = in_channel

    def forward(self, input):
        batch, channel, height, width = input.shape
        n_head = self.n_head
        head_dim = channel // n_head

        norm = self.norm(input)
        qkv = self.qkv(norm).reshape((batch, n_head, head_dim * 3, height, width))
        query, key, value = qkv.split(3, 2)  # bhdyx

        # attn = paddle.einsum(
        #     "bnchw, bncyx -> bnhwyx", query, key
        # ) / math.sqrt(channel)
        attn = query.reshape([0,0,0,-1]).transpose([0,1,3,2]).matmul(
            key.reshape([0,0,0,-1])
        ).reshape(query.shape[:2] + query.shape[-2:] + key.shape[-2:]) / math.sqrt(channel)
        attn = attn.reshape((batch, n_head, height, width, -1))
        attn = F.softmax(attn, -1)
        attn = attn.reshape((batch, n_head, height, width, height, width))

        # out = paddle.einsum("bnhwyx, bncyx -> bnchw", attn, value)
        out = value.reshape([0,0,0,-1]).matmul(
            attn.reshape([0,0,-1,attn.shape[-2]*attn.shape[-1]]).transpose([0,1,3,2])
        ).reshape(value.shape[:3] + attn.shape[2:4])
        out = self.out(out.reshape((batch, channel, height, width)))

        return out + input


class LinearSelfAttention(VanillaSelfAttention):
    def forward(self, input):
        batch, channel, height, width = input.shape
        n_head = self.n_head
        head_dim = channel // n_head

        norm = self.norm(input)
        qkv = self.qkv(norm).reshape((batch, n_head, head_dim * 3, height * width))
        q, k, v = qkv.split(3, 2)  # bhdn
        k = F.softmax(k, -1)

        # context = paddle.einsum('bhdn,bhen->bhde', k, v)
        context = paddle.matmul(k, v.transpose([0,1,3,2]))

        # out = paddle.einsum('bhdn,bhde->bhen', q, context)
        out = paddle.matmul(context.transpose([0,1,3,2]), q) 
        out = self.out(out.reshape((batch, channel, height, width)))

        return out + input


class TimeEmbedding(nn.Layer):
    def __init__(self, dim):
        super().__init__()

        self.dim = dim

        inv_freq = paddle.exp(
            paddle.arange(0, dim, 2, dtype=np.float32) * (-math.log(10000) / dim)
        )

        self.register_buffer("inv_freq", inv_freq)

    def forward(self, input):
        shape = input.shape
        sinusoid_in = input.reshape([-1,1]).astype('float32') * self.inv_freq.reshape([1,-1])
        pos_emb = paddle.concat([sinusoid_in.sin(), sinusoid_in.cos()], -1)
        pos_emb = pos_emb.reshape((*shape, self.dim))

        return pos_emb


class ResBlockWithAttention(nn.Layer):
    def __init__(
        self,
        in_channel,
        out_channel,
        time_dim,
        dropout,
        use_attention=False,
        attention_type='vanilla',
        attention_head=1,
        use_affine_time=False,
    ):
        super().__init__()

        self.resblocks = ResBlock(
            in_channel, out_channel, time_dim, use_affine_time, dropout
        )

        if use_attention:
            self.attention = (
                LinearSelfAttention if attention_type == 'linear' else VanillaSelfAttention
            )(out_channel, n_head=attention_head)

        else:
            self.attention = None

    def forward(self, input, time):
        out = self.resblocks(input, time)

        if self.attention is not None:
            out = self.attention(out)

        return out


def spatial_fold(input, fold):
    if fold == 1:
        return input

    batch, channel, height, width = input.shape
    h_fold = height // fold
    w_fold = width // fold

    return (
        input.reshape((batch, channel, h_fold, fold, w_fold, fold))
        .transpose((0, 1, 3, 5, 2, 4))
        .reshape((batch, -1, h_fold, w_fold))
    )


def spatial_unfold(input, unfold):
    if unfold == 1:
        return input

    batch, channel, height, width = input.shape
    h_unfold = height * unfold
    w_unfold = width * unfold

    return (
        input.reshape((batch, -1, unfold, unfold, height, width))
        .transpose((0, 1, 4, 2, 5, 3))
        .reshape((batch, -1, h_unfold, w_unfold))
    )


class UNet(nn.Layer):
    def __init__(
        self,
        in_channel,
        channel,
        channel_multiplier,
        n_res_blocks,
        attn_type = 'vanilla',
        attn_strides = [],
        attn_heads = 1,
        use_affine_time = False,
        dropout = 0,
        fold = 1,
    ):
        super().__init__()

        self.fold = fold

        time_dim = channel * 4

        n_block = len(channel_multiplier)

        self.time = nn.Sequential(
            TimeEmbedding(channel),
            linear(channel, time_dim),
            Swish(),
            linear(time_dim, time_dim),
        )

        down_layers = [conv2d(in_channel * (fold ** 2), channel, 3, padding=1)]
        feat_channels = [channel]
        in_channel = channel
        for i in range(n_block):
            for _ in range(n_res_blocks):
                channel_mult = channel * channel_multiplier[i]

                down_layers.append(
                    ResBlockWithAttention(
                        in_channel,
                        channel_mult,
                        time_dim,
                        dropout,
                        use_attention=attn_type != 'none' and 2 ** i in attn_strides,
                        attention_type=attn_type,
                        attention_head=attn_heads,
                        use_affine_time=use_affine_time,
                    )
                )

                feat_channels.append(channel_mult)
                in_channel = channel_mult

            if i != n_block - 1:
                down_layers.append(Downsample(in_channel))
                feat_channels.append(in_channel)

        self.down = nn.LayerList(down_layers)

        self.mid = nn.LayerList(
            [
                ResBlockWithAttention(
                    in_channel,
                    in_channel,
                    time_dim,
                    dropout=dropout,
                    use_attention=attn_type != 'none',
                    attention_type=attn_type,
                    attention_head=attn_heads,
                    use_affine_time=use_affine_time,
                ),
                ResBlockWithAttention(
                    in_channel,
                    in_channel,
                    time_dim,
                    dropout=dropout,
                    use_affine_time=use_affine_time,
                ),
            ]
        )

        up_layers = []
        for i in reversed(range(n_block)):
            for _ in range(n_res_blocks + 1):
                channel_mult = channel * channel_multiplier[i]

                up_layers.append(
                    ResBlockWithAttention(
                        in_channel + feat_channels.pop(),
                        channel_mult,
                        time_dim,
                        dropout=dropout,
                        use_attention=attn_type != 'none' and 2 ** i in attn_strides,
                        attention_type=attn_type,
                        attention_head=attn_heads,
                        use_affine_time=use_affine_time,
                    )
                )

                in_channel = channel_mult

            if i != 0:
                up_layers.append(Upsample(in_channel))

        self.up = nn.LayerList(up_layers)

        self.out = nn.Sequential(
            nn.GroupNorm(32, in_channel),
            Swish(),
            conv2d(in_channel, 3 * (fold ** 2), 3, padding=1, scale=1e-10),
        )

    def forward(self, input, time):
        time_embed = self.time(time)

        feats = []

        out = spatial_fold(input, self.fold)
        for layer in self.down:
            if isinstance(layer, ResBlockWithAttention):
                out = layer(out, time_embed)

            else:
                out = layer(out)

            feats.append(out)

        for layer in self.mid:
            out = layer(out, time_embed)

        for layer in self.up:
            if isinstance(layer, ResBlockWithAttention):
                out = layer(paddle.concat((out, feats.pop()), 1), time_embed)

            else:
                out = layer(out)

        out = self.out(out)
        out = spatial_unfold(out, self.fold)

        return out
