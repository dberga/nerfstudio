# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Ha-NeRF field"""


from typing import Dict, Optional, Tuple, Type

import torch
from torch import Tensor, nn

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.field_components.encodings import Encoding, Identity
from nerfstudio.field_components.field_heads import (
    DensityFieldHead,
    FieldHead,
    FieldHeadNames,
    RGBFieldHead,
)
from nerfstudio.field_components.mlp import MLP
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.base_field import Field

from nerfstudio.field_components.hallucidated import E_attr, implicit_mask #other: PosEmbedding

class HaNeRFField(Field):
    """Ha-NeRF Field

    Args:
        position_encoding: Position encoder.
        direction_encoding: Direction encoder.
        base_mlp_num_layers: Number of layers for base MLP.
        base_mlp_layer_width: Width of base MLP layers.
        head_mlp_num_layers: Number of layer for output head MLP.
        head_mlp_layer_width: Width of output head MLP layers.
        skip_connections: Where to add skip connection in base MLP.
        use_integrated_encoding: Used integrated samples as encoding input.
        spatial_distortion: Spatial distortion.
    """

    def __init__(
        self,
        position_encoding: Encoding = Identity(in_dim=3),
        direction_encoding: Encoding = Identity(in_dim=3),
        base_mlp_num_layers: int = 8,
        base_mlp_layer_width: int = 256,
        head_mlp_num_layers: int = 2,
        head_mlp_layer_width: int = 128,
        skip_connections: Tuple[int] = (4,),
        field_heads: Optional[Tuple[Type[FieldHead]]] = (RGBFieldHead,),
        use_integrated_encoding: bool = False,
        spatial_distortion: Optional[SpatialDistortion] = None,
        
        #################### Ha-Nerf default params
        use_mask: bool = True,
        encode_a: bool = True,
        appearance_encoding: Encoding = E_attr(3, 48),
        mask_embedding = implicit_mask()
        ####################
    ) -> None:
        super().__init__()
        self.position_encoding = position_encoding
        self.direction_encoding = direction_encoding
        self.use_integrated_encoding = use_integrated_encoding
        self.appearance_encoding = appearance_encoding
        self.mask_embedding = mask_embedding

        self.mlp_base = MLP(
            in_dim=self.position_encoding.get_out_dim(),
            num_layers=base_mlp_num_layers,
            layer_width=base_mlp_layer_width,
            skip_connections=skip_connections,
            out_activation=nn.ReLU(),
        )

        self.mlp_head = MLP(
            in_dim=self.mlp_base.get_out_dim() + self.direction_encoding.get_out_dim(),
            num_layers=head_mlp_num_layers,
            layer_width=head_mlp_layer_width,
            out_activation=nn.ReLU(),
        )

        '''
        To do: remake MLP field structure according to Ha-Nerf encodings

        # overall definitions (XingYu Ha-Nerf class)
        https://github.com/rover-xingyu/Ha-NeRF/blob/main/train_mask_grid_sample.py#L36
        # forward usage (training)
        https://github.com/rover-xingyu/Ha-NeRF/blob/main/train_mask_grid_sample.py#L121
        https://github.com/rover-xingyu/Ha-NeRF/blob/main/train_mask_grid_sample.py#L128
        # inference usage (eval)
        https://github.com/rover-xingyu/Ha-NeRF/blob/main/eval.py#L147
        # rendering inference with appearance encoding
        https://github.com/rover-xingyu/Ha-NeRF/blob/main/models/rendering.py#L112
        '''

        self.field_output_density = DensityFieldHead(in_dim=self.mlp_base.get_out_dim())
        self.field_heads = nn.ModuleList([field_head() for field_head in field_heads] if field_heads else [])  # type: ignore
        for field_head in self.field_heads:
            field_head.set_in_dim(self.mlp_head.get_out_dim())  # type: ignore

    def get_density(self, ray_samples: RaySamples) -> Tuple[Tensor, Tensor]:

        '''
        To do: include encoding according to Ha-Nerf

        # overall definitions (XingYu Ha-Nerf class)
        https://github.com/rover-xingyu/Ha-NeRF/blob/main/train_mask_grid_sample.py#L36
        # forward usage (training)
        https://github.com/rover-xingyu/Ha-NeRF/blob/main/train_mask_grid_sample.py#L121
        https://github.com/rover-xingyu/Ha-NeRF/blob/main/train_mask_grid_sample.py#L128
        # inference usage (eval)
        https://github.com/rover-xingyu/Ha-NeRF/blob/main/eval.py#L147
        # rendering inference with appearance encoding
        https://github.com/rover-xingyu/Ha-NeRF/blob/main/models/rendering.py#L112
        '''

        if self.use_integrated_encoding:
            gaussian_samples = ray_samples.frustums.get_gaussian_blob()
            if self.spatial_distortion is not None:
                gaussian_samples = self.spatial_distortion(gaussian_samples)
            encoded_xyz = self.position_encoding(gaussian_samples.mean, covs=gaussian_samples.cov)
        else:
            positions = ray_samples.frustums.get_positions()
            if self.spatial_distortion is not None:
                positions = self.spatial_distortion(positions)
            encoded_xyz = self.position_encoding(positions)
        base_mlp_out = self.mlp_base(encoded_xyz)
        density = self.field_output_density(base_mlp_out)
        return density, base_mlp_out

    def get_outputs( 
        self, ray_samples: RaySamples, density_embedding: Optional[Tensor] = None
    ) -> Dict[FieldHeadNames, Tensor]:

        '''
        To do: remake outputs according to Ha-Nerf

        # overall definitions (XingYu Ha-Nerf class)
        https://github.com/rover-xingyu/Ha-NeRF/blob/main/train_mask_grid_sample.py#L36
        # forward usage (training)
        https://github.com/rover-xingyu/Ha-NeRF/blob/main/train_mask_grid_sample.py#L121
        https://github.com/rover-xingyu/Ha-NeRF/blob/main/train_mask_grid_sample.py#L128
        # inference usage (eval)
        https://github.com/rover-xingyu/Ha-NeRF/blob/main/eval.py#L147
        # rendering inference with appearance encoding
        https://github.com/rover-xingyu/Ha-NeRF/blob/main/models/rendering.py#L112
        '''

        outputs = {}
        for field_head in self.field_heads:
            encoded_dir = self.direction_encoding(ray_samples.frustums.directions)
            mlp_out = self.mlp_head(torch.cat([encoded_dir, density_embedding], dim=-1))  # type: ignore
            outputs[field_head.field_head_name] = field_head(mlp_out)
        return outputs
