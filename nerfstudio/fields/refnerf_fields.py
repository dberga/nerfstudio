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

"""
Implementation of ref-NeRF based on mip-Nerf.
"""

from typing import Dict, Optional, Tuple, Type, Literal

import numpy as np
import math
import torch
from torch import Tensor, nn
from torch.nn.parameter import Parameter

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.field_components.encodings import Encoding, Identity
from nerfstudio.field_components.field_heads import (
    DensityFieldHead,
    PredNormalsFieldHead,
    FieldHead,
    FieldHeadNames,
    RGBFieldHead,
)
from nerfstudio.field_components.mod_field_heads import (
    RoughnessFieldHead,
    SpecularTintFieldHead,
)
from nerfstudio.field_components.mlp import MLP
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.fields.base_field import Field, get_normalized_directions


class RefNeRFField(Field):
    """NeRF Field

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
        hidden_dim_transient: dimension of hidden layers for transient network
        geo_feat_dim: output geo feat dimensions
        use_pred_normals: whether to use predicted normals
        use_pred_specular_tint: whether to use predicted specular tint
    """

    def __init__(
        self,
        aabb: Tensor,
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
        hidden_dim_transient: int = 64,
        geo_feat_dim: int = 15,
        use_pred_normals: bool = True,
        use_pred_specular_tint: bool = True,
        implementation: Literal["tcnn", "torch"] = "tcnn",
    ) -> None:
        super().__init__()
        self.aabb = Parameter(aabb, requires_grad=False)
        self.geo_feat_dim = geo_feat_dim
        
        self.position_encoding = position_encoding
        self.direction_encoding = direction_encoding
        self.use_integrated_encoding = use_integrated_encoding
        self.spatial_distortion = spatial_distortion
        self.use_pred_normals = use_pred_normals
        self.use_pred_specular_tint = use_pred_specular_tint
        #self.roughness_bias: float = -1.0

        self.mlp_density = MLP(
            in_dim=self.position_encoding.get_out_dim(),
            num_layers=base_mlp_num_layers,
            layer_width=base_mlp_layer_width,
            skip_connections=skip_connections,
            out_activation=nn.ReLU(),
        )

        # diffuse color
        self.mlp_diffuse_color = MLP(
            in_dim=self.position_encoding.get_out_dim(),
            num_layers=base_mlp_num_layers,
            layer_width=base_mlp_layer_width,
            skip_connections=skip_connections,
            out_activation=nn.ReLU(),
        )

        # roughness
        self.mlp_roughness = MLP(
            in_dim=self.position_encoding.get_out_dim(),
            num_layers=base_mlp_num_layers,
            layer_width=base_mlp_layer_width,
            skip_connections=skip_connections,
            out_activation=nn.ReLU(),
        )
        self.geo_feat_dim = self.mlp_density.get_out_dim() + self.direction_encoding.get_out_dim() # wrong dims here?

        # specular tint
        self.mlp_specular_tint = MLP(
            in_dim=self.geo_feat_dim + self.position_encoding.get_out_dim(), # or wrong dims here?
            num_layers=3,
            layer_width=64,
            out_dim=hidden_dim_transient,
            activation=nn.ReLU(),
            out_activation=None,
            implementation=implementation,
            )
        
        # normals
        self.mlp_pred_normals = MLP(
            in_dim=self.geo_feat_dim + self.position_encoding.get_out_dim(),
            num_layers=3,
            layer_width=64,
            out_dim=hidden_dim_transient,
            activation=nn.ReLU(),
            out_activation=None,
            implementation=implementation,
        )

        self.mlp_directional_head = MLP(
            in_dim=self.mlp_density.get_out_dim() + self.direction_encoding.get_out_dim(),
            num_layers=head_mlp_num_layers,
            layer_width=head_mlp_layer_width,
            out_activation=nn.ReLU(),
        )

        self.field_output_density = DensityFieldHead(in_dim=self.mlp_density.get_out_dim())
        self.field_output_diffuse_color = RGBFieldHead(in_dim=self.mlp_diffuse_color.get_out_dim())
        self.field_output_specular_tint = SpecularTintFieldHead(in_dim=self.mlp_specular_tint.get_out_dim())
        self.field_output_roughness = RoughnessFieldHead(in_dim=self.mlp_roughness.get_out_dim(), activation=nn.Sigmoid())
        self.field_head_pred_normals = PredNormalsFieldHead(in_dim=self.mlp_pred_normals.get_out_dim(), activation=nn.Sigmoid())
        
        self.field_heads = nn.ModuleList([field_head() for field_head in field_heads] if field_heads else [])  # type: ignore
        #for field_head in self.field_heads:
        #    field_head.set_in_dim(self.mlp_density.get_out_dim())  # type: ignore

    def get_density(self, ray_samples: RaySamples) -> Tuple[Tensor, Tensor]:
        try:
            if self.spatial_distortion is not None:
                positions = ray_samples.frustums.get_positions()
                positions = self.spatial_distortion(positions)
                positions = (positions + 2.0) / 4.0
            else:
                positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)

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

            # Make sure the tcnn gets inputs between 0 and 1.
            #selector = ((positions > 0.0) & (positions < 1.0)).all(dim=-1)
            #positions = positions * selector[..., None]
            self._sample_locations = positions
            if not self._sample_locations.requires_grad:
                self._sample_locations.requires_grad = True
            #positions_flat = positions.view(-1, 3)
            #h = self.mlp_base(positions_flat).view(*ray_samples.frustums.shape, -1)
            #density_before_activation, aux_base_mlp_out = torch.split(h, [1, self.geo_feat_dim], dim=-1)
            #self._density_before_activation = density_before_activation
            base_mlp_out = self.mlp_density(encoded_xyz)
            self._density_before_activation = base_mlp_out
            density = self.field_output_density(base_mlp_out)
        except:
            import pdb; pdb.set_trace()
        return density, base_mlp_out

    """def get_density(self, ray_samples: RaySamples) -> Tuple[Tensor, Tensor]:
        encoded_xyz = self.positional_encoding(ray_samples)
        density_mlp_out = self.mlp_density(encoded_xyz)
        density = self.field_output_density(density_mlp_out)
        return density, density_mlp_out"""

    def get_diffuse_color(self, ray_samples: RaySamples) -> Tensor:
        encoded_xyz = self.positional_encoding(ray_samples)
        diffuse_color_mlp_out = self.mlp_diffuse_color(encoded_xyz)
        return self.field_output_diffuse_color(diffuse_color_mlp_out)
    
    def get_specular_tint(self, ray_samples: RaySamples) -> Tensor:
        try:
            encoded_xyz = self.positional_encoding(ray_samples)
            specular_tint_mlp_out = self.mlp_specular_tint(encoded_xyz)
        except:
            import pdb; pdb.set_trace()
        return self.field_output_specular_tint(specular_tint_mlp_out)
    
    def get_roughness(self, ray_samples: RaySamples) -> Tensor:
        encoded_xyz = self.positional_encoding(ray_samples)
        roughness_mlp_out = self.get_roughness(encoded_xyz)
        return self.field_output_specular_tint(roughness_mlp_out)

    def get_outputs(
        self, ray_samples: RaySamples, density_embedding: Optional[Tensor] = None
    ) -> Dict[FieldHeadNames, Tensor]:
        assert density_embedding is not None
        directions = get_normalized_directions(ray_samples.frustums.directions)
        outputs_shape = ray_samples.frustums.directions.shape[:-1]
        outputs = {}

        """for field_head in self.field_heads:
            encoded_dir = self.direction_encoding(ray_samples.frustums.directions)
            mlp_out = self.mlp_head(torch.cat([encoded_dir, density_embedding], dim=-1))  # type: ignore
            outputs[field_head.field_head_name] = field_head(mlp_out)
        """

        ######
        # Spatial MLP

        # diffuse color 
        outputs[FieldHeadNames.DIFFUSE_COLOR] = self.get_diffuse_color(ray_samples)

        # specular tint
        outputs[FieldHeadNames.SPECULAR_TINT] = self.get_specular_tint(ray_samples)

        # roughness
        outputs[FieldHeadNames.ROUGHNESS] = self.get_specular_tint(ray_samples)

        # predicted normals
        positions = ray_samples.frustums.get_positions()

        positions_flat = self.position_encoding(positions.view(-1, 3))
        pred_normals_inp = torch.cat([positions_flat, density_embedding.view(-1, self.geo_feat_dim)], dim=-1)

        x = self.mlp_pred_normals(pred_normals_inp).view(*outputs_shape, -1).to(directions)
        outputs[FieldHeadNames.PRED_NORMALS] = self.field_head_pred_normals(x)
        ######


        reflection_vector = RefNeRFField.reflect(directions, outputs[FieldHeadNames.PRED_NORMALS])
        dot_product = torch.dot(directions, outputs[FieldHeadNames.PRED_NORMALS])
        #ide = RefNeRFField.generate_ide_fn()

        ######
        # Directional MLP

        # specular color

        ######
            
        return outputs
    
    def positional_encoding(self, ray_samples):
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
        return encoded_xyz

    @staticmethod
    def reflect(viewdirs, normals):
        """Reflect view directions about normals.

        The reflection of a vector v about a unit vector n is a vector u such that
        dot(v, n) = dot(u, n), and dot(u, u) = dot(v, v). The solution to these two
        equations is u = 2 dot(n, v) n - v.

        Args:
            viewdirs: [..., 3] array of view directions.
            normals: [..., 3] array of normal directions (assumed to be unit vectors).

        Returns:
            [..., 3] array of reflection directions.
        """
        return 2.0 * torch.sum(normals * viewdirs, dim=-1) * normals - viewdirs

    @staticmethod
    def generate_ide_fn(deg_view):
        """Generate integrated directional encoding (IDE) function.

        This function returns a function that computes the integrated directional
        encoding from Equations 6-8 of arxiv.org/abs/2112.03907.

        Args:
            deg_view: number of spherical harmonics degrees to use.

        Returns:
            A function for evaluating integrated directional encoding.

        Raises:
            ValueError: if deg_view is larger than 5.
        """
        if deg_view > 5:
            raise ValueError("Only deg_view of at most 5 is numerically stable.")

        ml_array = get_ml_array(deg_view)
        l_max = 2 ** (deg_view - 1)

        # Create a matrix corresponding to ml_array holding all coefficients, which,
        # when multiplied (from the right) by the z coordinate Vandermonde matrix,
        # results in the z component of the encoding.
        mat = np.zeros((l_max + 1, ml_array.shape[1]))
        for i, (m, l) in enumerate(ml_array.T):
            for k in range(l - m + 1):
                mat[k, i] = sph_harm_coeff(l, m, k)

        mat = torch.Tensor(mat)

        def integrated_dir_enc_fn(xyz, kappa_inv):
            """Function returning integrated directional encoding (IDE).

            Args:
                xyz: [..., 3] array of Cartesian coordinates of directions to evaluate at.
                kappa_inv: [..., 1] reciprocal of the concentration parameter of the von
                    Mises-Fisher distribution.

            Returns:
                An array with the resulting IDE.
            """
            x = xyz[..., 0:1]
            y = xyz[..., 1:2]
            z = xyz[..., 2:3]

            vmz = torch.cat([z**i for i in range(mat.shape[0])], dim=-1)
            vmxy = torch.cat([(x + 1j * y) ** m for m in ml_array[0, :]], dim=-1)

            sph_harms = vmxy * torch.matmul(vmz, mat.to(xyz.device))

            sigma = torch.Tensor(0.5 * ml_array[1, :] * (ml_array[1, :] + 1)).to(kappa_inv.device)
            ide = sph_harms * torch.exp(-sigma * kappa_inv)

            return torch.cat([torch.real(ide), torch.imag(ide)], dim=-1)

        return integrated_dir_enc_fn
    
def get_ml_array(deg_view):
    """Create a list with all pairs of (l, m) values to use in the encoding."""
    ml_list = []
    for i in range(deg_view):
        l = 2**i
        # Only use nonnegative m values, later splitting real and imaginary parts.
        for m in range(l + 1):
            ml_list.append((m, l))

    ml_array = np.array(ml_list).T
    return ml_array

def assoc_legendre_coeff(l, m, k):
    """Compute associated Legendre polynomial coefficients.

    Returns the coefficient of the cos^k(theta)*sin^m(theta) term in the
    (l, m)th associated Legendre polynomial, P_l^m(cos(theta)).

    Args:
        l: associated Legendre polynomial degree.
        m: associated Legendre polynomial order.
        k: power of cos(theta).

    Returns:
        A float, the coefficient of the term corresponding to the inputs.
    """
    return (
        (-1) ** m
        * 2**l
        * math.factorial(l)
        / math.factorial(k)
        / math.factorial(l - k - m)
        * generalized_binomial_coeff(0.5 * (l + k + m - 1.0), l)
    )

def generalized_binomial_coeff(a, k):
    """Compute generalized binomial coefficients."""
    # return np.prod(a - np.arange(k)) / np.math.factorial(k)
    return np.prod(a - np.arange(k)) / math.factorial(k)


def sph_harm_coeff(l, m, k):
    """Compute spherical harmonic coefficients."""
    # return (np.sqrt(
    #     (2.0 * l + 1.0) * np.math.factorial(l - m) /
    #     (4.0 * np.pi * np.math.factorial(l + m))) * assoc_legendre_coeff(l, m, k))
    return np.sqrt(
        (2.0 * l + 1.0) * math.factorial(l - m) / (4.0 * np.pi * math.factorial(l + m))
    ) * assoc_legendre_coeff(l, m, k)
