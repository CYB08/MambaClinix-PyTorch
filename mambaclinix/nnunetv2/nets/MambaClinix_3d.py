import numpy as np
import math
import torch
from torch import nn
from torch.nn import functional as F
from typing import Union, Type, List, Tuple

from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim

from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from dynamic_network_architectures.building_blocks.helper import get_matching_instancenorm, convert_dim_to_conv_op
from nnunetv2.utilities.network_initialization import InitWeights_He
from mamba_ssm import Mamba
from dynamic_network_architectures.building_blocks.helper import maybe_convert_scalar_to_list, get_matching_pool_op
from torch.cuda.amp import autocast
from dynamic_network_architectures.building_blocks.simple_conv_blocks import ConvDropoutNormReLU
from dynamic_network_architectures.building_blocks.regularization import DropPath
from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks
from dynamic_network_architectures.building_blocks.residual import StackedResidualBlocks
from dynamic_network_architectures.building_blocks.residual import BasicBlockD, BottleneckD


class LayerNorm(nn.Module):
    # Layer normalization for D, W, H
    def __init__(self, input_channels, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(input_channels))
        self.beta = nn.Parameter(torch.zeros(input_channels))
        self.input_channels = input_channels
        self.eps = eps

    def forward(self, x):
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.input_channels,), self.gamma, self.beta, self.eps)

        return x.transpose(1, -1)


class hgConv(nn.Module):
    r""" high-order gated convolution """
    def __init__(self,
                 conv_op: Type[_ConvNd],
                 input_channels: int,
                 output_channels: int,
                 kernel_size: Union[int, List[int], Tuple[int, ...]],
                 stride: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 order: int = 5,
                 s: float = 1,
                 ):
        super().__init__()

        self.scale = s
        self.order = order
        self.dims = [input_channels // 2 ** i for i in range(order)]
        self.dims.reverse()

        kernel_size = maybe_convert_scalar_to_list(conv_op, kernel_size)
        stride = maybe_convert_scalar_to_list(conv_op, stride)

        self.proj_in = conv_op(input_channels, 2 * input_channels,
                               (1, 1, 1),
                               stride,
                               padding=[(i - 1) // 2 for i in (1, 1, 1)],
                               dilation=1,
                               bias=conv_bias
                               )

        self.dwconv = conv_op(sum(self.dims),
                              sum(self.dims),
                              kernel_size,
                              stride,
                              padding=[(i - 1) // 2 for i in kernel_size],
                              dilation=1,
                              bias=conv_bias,
                              groups=sum(self.dims)
                              )

        self.reconv = nn.ModuleList([conv_op(self.dims[i], self.dims[i + 1], (1, 1, 1)) for i in range(order - 1)])

        self.proj_out = conv_op(input_channels, output_channels, (1, 1, 1))

    def forward(self, x):
        split_x = self.proj_in(x)
        basex, expanded_x = torch.split(split_x, (self.dims[0], sum(self.dims)), dim=1)
        ex = self.dwconv(expanded_x) * self.scale
        ex_list = torch.split(ex, self.dims, dim=1)
        x = basex + ex_list[0]
        for i in range(self.order - 1):
            x = self.reconv[i](x) + ex_list[i + 1]

        return self.proj_out(x)


class HGCN_BLOCK(nn.Module):
    r""" hierarchical gated convolutional block """
    def __init__(self,
                 conv_op: Type[_ConvNd],
                 input_channels: int,
                 output_channels: int,
                 transform_channels: int,
                 kernel_size: Union[int, List[int], Tuple[int, ...]],
                 stride: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 layer_eps: float = 1e-6,
                 drop_path: int = 0,
                 hgConvs = hgConv,
                 order: int = 3,
                 ):

        super().__init__()

        kernel_size = maybe_convert_scalar_to_list(conv_op, kernel_size)
        stride = maybe_convert_scalar_to_list(conv_op, stride)

        # high-order conv
        self.hgConvs = hgConvs(conv_op,
                               transform_channels,
                               transform_channels,
                               kernel_size,
                               stride,
                               conv_bias,
                               order,
                               )

        self.lynorm1 = LayerNorm(transform_channels, eps=layer_eps)
        self.lynorm2 = LayerNorm(transform_channels, eps=layer_eps)

        if norm_op_kwargs is None:
            norm_op_kwargs = {}

        if nonlin_kwargs is None:
            nonlin_kwargs = {}

        self.transform_channels = transform_channels

        self.transconv = ConvDropoutNormReLU(conv_op, input_channels, transform_channels, 1, 1, conv_bias)

        self.conv1 = ConvDropoutNormReLU(conv_op, transform_channels, output_channels, kernel_size, stride, conv_bias,
                                         None, None, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs)

        self.conv2 = ConvDropoutNormReLU(conv_op, output_channels, output_channels, kernel_size, stride, conv_bias)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.act = nonlin(**nonlin_kwargs) if nonlin is not None else lambda x: x

        self.gamma1 = nn.Parameter(layer_eps * torch.ones(transform_channels),
                                   requires_grad=True) if layer_eps > 0 else None

    def forward(self, x):
        gamma1 = self.gamma1.view(self.transform_channels, 1, 1, 1)
        skip = x
        x = self.transconv(x)
        x = x + self.drop_path(gamma1 * self.hgConvs(self.lynorm1(x)))
        out = self.conv2(self.conv1(self.lynorm2(x)))

        return skip + self.drop_path(out)  # self.act(skip + self.drop_path(out))


class UpsampleLayer(nn.Module):
    def __init__(
            self,
            conv_op,
            input_channels,
            output_channels,
            pool_op_kernel_size,
            mode='nearest'
    ):
        super().__init__()
        self.conv = conv_op(input_channels, output_channels, kernel_size=1)
        self.pool_op_kernel_size = pool_op_kernel_size
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.pool_op_kernel_size, mode=self.mode)
        x = self.conv(x)
        return x


class MambaLayer(nn.Module):
    def __init__(self, input_dim, output_dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        print(f"MambaLayer: input_dim: {input_dim}")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm = nn.LayerNorm(input_dim)
        self.mamba = Mamba(
            d_model=input_dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
        )
        self.proj = nn.Linear(input_dim, output_dim)
        self.skip_scale = nn.Parameter(torch.ones(1))

    def forward_res(self, x):
        B, d_model = x.shape[:2]
        x_skip = x
        assert d_model == self.input_dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, d_model, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        x_mamba = self.norm(x_mamba)
        x_mamba = self.proj(x_mamba)
        out = x_mamba.transpose(-1, -2).reshape(B, d_model, *img_dims)

        return out + x_skip

    @autocast(enabled=False)
    def forward(self, x):
        if x.dtype == torch.float16 or x.dtype == torch.bfloat16:
            x = x.type(torch.float32)
            out = self.forward_res(x)

        return out


class BasicResBlock(nn.Module):
    def __init__(
            self,
            conv_op,
            input_channels,
            output_channels,
            norm_op,
            norm_op_kwargs,
            kernel_size=3,
            padding=1,
            stride=1,
            use_1x1conv=False,
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={'inplace': True}
    ):
        super().__init__()

        self.conv1 = conv_op(input_channels, output_channels, kernel_size, stride=stride, padding=padding)
        self.norm1 = norm_op(output_channels, **norm_op_kwargs)
        self.act1 = nonlin(**nonlin_kwargs)

        self.conv2 = conv_op(output_channels, output_channels, kernel_size, padding=padding)
        self.norm2 = norm_op(output_channels, **norm_op_kwargs)
        self.act2 = nonlin(**nonlin_kwargs)

        if use_1x1conv:
            self.conv3 = conv_op(input_channels, output_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None

    def forward(self, x):
        y = self.conv1(x)
        y = self.act1(self.norm1(y))
        y = self.norm2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        y += x
        return self.act2(y)


class ResidualMambaEncoder(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...], Tuple[Tuple[int, ...], ...]],
                 n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 block: Union[Type[BasicBlockD], Type[BottleneckD]] = BasicBlockD,
                 bottleneck_channels: Union[int, List[int], Tuple[int, ...]] = None,
                 return_skips: bool = False,
                 stem_channels: int = None,
                 pool_type: str = 'conv',
                 drop_path_rate = 0.5,
                 stochastic_depth_p: float = 0.0,
                 squeeze_excitation: bool = False,
                 squeeze_excitation_reduction_ratio: float = 1. / 16,
                 hgConv = hgConv,
                 HGCN_BLOCK = HGCN_BLOCK,
                 ):

        super().__init__()
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(strides, int):
            strides = [strides] * n_stages
        if bottleneck_channels is None or isinstance(bottleneck_channels, int):
            bottleneck_channels = [bottleneck_channels] * n_stages
        if not isinstance(hgConv, list):
            hgConv_list = [hgConv] * n_stages
        assert len(
            bottleneck_channels) == n_stages, "bottleneck_channels must be None or have as many entries as we have resolution stages (n_stages)"
        assert len(
            kernel_sizes) == n_stages, "kernel_sizes must have as many entries as we have resolution stages (n_stages)"
        assert len(
            n_blocks_per_stage) == n_stages, "n_conv_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert len(
            features_per_stage) == n_stages, "features_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert len(strides) == n_stages, "strides must have as many entries as we have resolution stages (n_stages). " \
                                         "Important: first entry is recommended to be 1, else we run strided conv drectly on the input"

        pool_op = get_matching_pool_op(conv_op, pool_type=pool_type) if pool_type != 'conv' else None

        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, n_stages)]

        # the spatial interaction orders for each HGCN block
        order_range = [temp_order + 1 for temp_order in range(1, n_stages + 1)]

        self.conv_pad_sizes = []
        for krnl in kernel_sizes:
            self.conv_pad_sizes.append([i // 2 for i in krnl])

        # stem function
        stem_channels = features_per_stage[0]
        self.stem = StackedConvBlocks(1, conv_op, input_channels, stem_channels, kernel_sizes[0], 1, conv_bias,
                                      norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs)
        input_channels = stem_channels

        stages = []
        encoder_layers = []
        for s in range(n_stages):
            stride_for_conv = strides[s] if pool_op is None else 1
            stage = StackedResidualBlocks(
                1, conv_op, input_channels, features_per_stage[s], kernel_sizes[s], stride_for_conv,
                conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs,
                block=block, bottleneck_channels=bottleneck_channels[s], stochastic_depth_p=stochastic_depth_p,
                squeeze_excitation=squeeze_excitation,
                squeeze_excitation_reduction_ratio=squeeze_excitation_reduction_ratio
            )

            if n_blocks_per_stage[s] == 2:
                encoder_layers.append(
                    HGCN_BLOCK(
                        conv_op=conv_op,
                        input_channels=features_per_stage[s],
                        output_channels=features_per_stage[s],
                        transform_channels=features_per_stage[s]//2, #stem_channels,
                        kernel_size=kernel_sizes[s],
                        stride=1,
                        conv_bias=conv_bias,
                        norm_op=norm_op,
                        norm_op_kwargs=norm_op_kwargs,
                        nonlin=nonlin,
                        nonlin_kwargs=nonlin_kwargs,
                        layer_eps=1e-6,
                        drop_path=dp_rates[s],
                        hgConvs=hgConv_list[s],
                        order=order_range[s],
                    )
                )

            elif n_blocks_per_stage[s] == 1:
                encoder_layers.append(
                    MambaLayer(input_dim=features_per_stage[s], output_dim=features_per_stage[s])
                )

            stages.append(stage)
            input_channels = features_per_stage[s]

        self.conn_pos = len(n_blocks_per_stage) - 1 - n_blocks_per_stage[::-1].index(2)
        self.LayerNorm = LayerNorm(features_per_stage[self.conn_pos], eps=1e-6)
        self.encoder_layers = nn.ModuleList(encoder_layers)
        self.stages = nn.ModuleList(stages)
        self.output_channels = features_per_stage
        self.strides = [maybe_convert_scalar_to_list(conv_op, i) for i in strides]
        self.return_skips = return_skips

        self.conv_op = conv_op
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.conv_bias = conv_bias
        self.kernel_sizes = kernel_sizes
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs

    def forward(self, x):
        if self.stem is not None:
            x = self.stem(x)
        ret = []
        for s in range(len(self.stages)):
            x = self.stages[s](x)
            if s == self.conn_pos:
                x = self.LayerNorm(x)
            x = self.encoder_layers[s](x)
            ret.append(x)
        if self.return_skips:
            return ret
        else:
            return ret[-1]

    def compute_conv_feature_map_size(self, input_size):
        if self.stem is not None:
            output = self.stem.compute_conv_feature_map_size(input_size)
        else:
            output = np.int64(0)

        for s in range(len(self.stages)):
            output += self.stages[s].compute_conv_feature_map_size(input_size)
            input_size = [i // j for i, j in zip(input_size, self.strides[s])]

        return output



class UNetResDecoder(nn.Module):
    def __init__(self,
                 encoder,
                 num_classes,
                 n_conv_per_stage: Union[int, Tuple[int, ...], List[int]],
                 deep_supervision, nonlin_first: bool = False):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder = encoder
        self.num_classes = num_classes
        n_stages_encoder = len(encoder.output_channels)
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)
        assert len(n_conv_per_stage) == n_stages_encoder - 1, "n_conv_per_stage must have as many entries as we have " \
                                                              "resolution stages - 1 (n_stages in encoder - 1), " \
                                                              "here: %d" % n_stages_encoder

        self.Init_Residual = BasicBlockD(
            conv_op=encoder.conv_op,
            input_channels=encoder.output_channels[-1],
            output_channels=encoder.output_channels[-1],
            kernel_size=encoder.kernel_sizes[-1],
            stride=1,
            conv_bias=encoder.conv_bias,
            norm_op=encoder.norm_op,
            norm_op_kwargs=encoder.norm_op_kwargs,
            nonlin=encoder.nonlin,
            nonlin_kwargs=encoder.nonlin_kwargs,
        )

        stages = []
        upsample_layers = []
        seg_layers = []
        residual_blocks = []
        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]
            stride_for_upsampling = encoder.strides[-s]

            upsample_layers.append(UpsampleLayer(
                conv_op=encoder.conv_op,
                input_channels=input_features_below,
                output_channels=input_features_skip,
                pool_op_kernel_size=stride_for_upsampling,
                mode='nearest'
            ))

            residual_blocks.append(
                BasicBlockD(
                    conv_op=encoder.conv_op,
                    input_channels=input_features_skip,
                    output_channels=input_features_skip,
                    kernel_size=encoder.kernel_sizes[-(s + 1)],
                    stride=1,
                    conv_bias=encoder.conv_bias,
                    norm_op=encoder.norm_op,
                    norm_op_kwargs=encoder.norm_op_kwargs,
                    nonlin=encoder.nonlin,
                    nonlin_kwargs=encoder.nonlin_kwargs,
                )
            )

            stages.append(
                BasicResBlock(
                    conv_op=encoder.conv_op,
                    norm_op=encoder.norm_op,
                    norm_op_kwargs=encoder.norm_op_kwargs,
                    nonlin=encoder.nonlin,
                    nonlin_kwargs=encoder.nonlin_kwargs,
                    input_channels=2 * input_features_skip,
                    output_channels=input_features_skip,
                    kernel_size=encoder.kernel_sizes[-(s + 1)],
                    padding=encoder.conv_pad_sizes[-(s + 1)],
                    stride=1,
                    use_1x1conv=True
                )
            )

            seg_layers.append(encoder.conv_op(input_features_skip, num_classes, 1, 1, 0, bias=True))

        self.residual_blocks = nn.ModuleList(residual_blocks)
        self.stages = nn.ModuleList(stages)
        self.upsample_layers = nn.ModuleList(upsample_layers)
        self.seg_layers = nn.ModuleList(seg_layers)

    def forward(self, skips):
        lres_input = skips[-1]
        lres_input = self.Init_Residual(lres_input)
        seg_outputs = []
        for s in range(len(self.stages)):
            x = self.upsample_layers[s](lres_input)
            res = self.residual_blocks[s](skips[-(s + 2)])
            x = torch.cat((x, res), 1)
            x = self.stages[s](x)
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            elif s == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[-1](x))
            lres_input = x

        seg_outputs = seg_outputs[::-1]

        if not self.deep_supervision:
            r = seg_outputs[0]
        else:
            r = seg_outputs
        return r

    def compute_conv_feature_map_size(self, input_size):
        skip_sizes = []
        for s in range(len(self.encoder.strides) - 1):
            skip_sizes.append([i // j for i, j in zip(input_size, self.encoder.strides[s])])
            input_size = skip_sizes[-1]

        assert len(skip_sizes) == len(self.stages)

        output = np.int64(0)
        for s in range(len(self.stages)):
            output += self.stages[s].compute_conv_feature_map_size(skip_sizes[-(s + 1)])
            output += np.prod([self.encoder.output_channels[-(s + 2)], *skip_sizes[-(s + 1)]], dtype=np.int64)
            if self.deep_supervision or (s == (len(self.stages) - 1)):
                output += np.prod([self.num_classes, *skip_sizes[-(s + 1)]], dtype=np.int64)
        return output


class MambaClinix(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False,
                 block: Union[Type[BasicBlockD], Type[BottleneckD]] = BasicBlockD,
                 bottleneck_channels: Union[int, List[int], Tuple[int, ...]] = None,
                 stem_channels: int = None
                 ):
        super().__init__()
        n_blocks_per_stage = n_conv_per_stage
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)

        for s in range(math.ceil(n_stages / 2), n_stages):
            n_blocks_per_stage[s] = 1

        for s in range(math.ceil((n_stages - 1) / 2 + 0.5), n_stages - 1):
            n_conv_per_stage_decoder[s] = 1

        assert len(n_blocks_per_stage) == n_stages, "n_blocks_per_stage must have as many entries as we have " \
                                                    f"resolution stages. here: {n_stages}. " \
                                                    f"n_blocks_per_stage: {n_blocks_per_stage}"
        assert len(n_conv_per_stage_decoder) == (n_stages - 1), "n_conv_per_stage_decoder must have one less entries " \
                                                                f"as we have resolution stages. here: {n_stages} " \
                                                                f"stages, so it should have {n_stages - 1} entries. " \
                                                                f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"
        self.encoder = ResidualMambaEncoder(
            input_channels,
            n_stages,
            features_per_stage,
            conv_op,
            kernel_sizes,
            strides,
            n_blocks_per_stage,
            conv_bias,
            norm_op,
            norm_op_kwargs,
            dropout_op,
            dropout_op_kwargs,
            nonlin,
            nonlin_kwargs,
            block,
            bottleneck_channels,
            return_skips=True,
            stem_channels=stem_channels
        )

        self.decoder = UNetResDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision)

    def forward(self, x):
        skips = self.encoder(x)
        return self.decoder(skips)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(
            self.encoder.conv_op), "just give the image size without color/feature channels or " \
                                   "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                   "Give input_size=(x, y(, z))!"
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(
            input_size)


def get_mambaclinix_3d_from_plans(
        plans_manager: PlansManager,
        dataset_json: dict,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        deep_supervision: bool = True
):
    """
    we may have to change this in the future to accommodate other plans -> network mappings

    num_input_channels can differ depending on whether we do cascade. Its best to make this info available in the
    trainer rather than inferring it again from the plans here.
    """
    num_stages = len(configuration_manager.conv_kernel_sizes)

    dim = len(configuration_manager.conv_kernel_sizes[0])
    conv_op = convert_dim_to_conv_op(dim)

    label_manager = plans_manager.get_label_manager(dataset_json)

    segmentation_network_class_name = 'MambaClinix'
    network_class = MambaClinix
    kwargs = {
        'MambaClinix': {
            'conv_bias': True,
            'norm_op': get_matching_instancenorm(conv_op),
            'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
            'dropout_op': None, 'dropout_op_kwargs': None,
            'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'inplace': True},
        }
    }

    conv_or_blocks_per_stage = {
        'n_conv_per_stage': configuration_manager.n_conv_per_stage_encoder,
        'n_conv_per_stage_decoder': configuration_manager.n_conv_per_stage_decoder
    }

    model = network_class(
        input_channels=num_input_channels,
        n_stages=num_stages,
        features_per_stage=[min(configuration_manager.UNet_base_num_features * 2 ** i,
                                configuration_manager.unet_max_num_features) for i in range(num_stages)],
        conv_op=conv_op,
        kernel_sizes=configuration_manager.conv_kernel_sizes,
        strides=configuration_manager.pool_op_kernel_sizes,
        num_classes=label_manager.num_segmentation_heads,
        deep_supervision=deep_supervision,
        **conv_or_blocks_per_stage,
        **kwargs[segmentation_network_class_name]
    )
    model.apply(InitWeights_He(1e-2))

    return model