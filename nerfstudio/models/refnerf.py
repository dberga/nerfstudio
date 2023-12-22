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
from __future__ import annotations

from typing import Dict, List, Tuple, Type, Any, Literal
from dataclasses import dataclass, field

import numpy as np
import math
import torch
from torch.nn import Parameter
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from nerfstudio.configs.config_utils import to_immutable_dict
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.field_components.encodings import NeRFEncoding
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.temporal_distortions import TemporalDistortionKind
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.fields.refnerf_fields import RefNeRFField
from nerfstudio.fields.vanilla_nerf_field import NeRFField
from nerfstudio.fields.nerfacto_field import NerfactoField
from nerfstudio.fields.density_fields import HashMLPDensityField
from nerfstudio.model_components.losses import MSELoss
from nerfstudio.model_components.ray_samplers import PDFSampler, UniformSampler, ProposalNetworkSampler
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    RGBRenderer,
    NormalsRenderer,
)
from nerfstudio.model_components.shaders import NormalsShader
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps, colors, misc
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation


########### https://github.com/kakaobrain/nerf-factory/blob/main/src/model/refnerf/ref_utils.py # nerf-factory -> refnerf
########### https://colab.research.google.com/github/nexuslrf/ENVIDR/blob/main/demo.ipynb # ENVIDR

def generalized_binomial_coeff(a, k):
    """Compute generalized binomial coefficients."""
    # return np.prod(a - np.arange(k)) / np.math.factorial(k)
    return np.prod(a - np.arange(k)) / math.factorial(k)


def assoc_legendre_coeff(l, m, k): # type: ignore
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


def sph_harm_coeff(l, m, k): # type: ignore
    """Compute spherical harmonic coefficients."""
    # return (np.sqrt(
    #     (2.0 * l + 1.0) * np.math.factorial(l - m) /
    #     (4.0 * np.pi * np.math.factorial(l + m))) * assoc_legendre_coeff(l, m, k))
    return np.sqrt(
        (2.0 * l + 1.0) * math.factorial(l - m) / (4.0 * np.pi * math.factorial(l + m))
    ) * assoc_legendre_coeff(l, m, k)


def get_ml_array(deg_view):
    """Create a list with all pairs of (l, m) values to use in the encoding."""
    ml_list = []
    for i in range(deg_view):
        l = 2**i # type: ignore
        # Only use nonnegative m values, later splitting real and imaginary parts.
        for m in range(l + 1):
            ml_list.append((m, l))

    ml_array = np.array(ml_list).T
    return ml_array


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
    for i, (m, l) in enumerate(ml_array.T): # type: ignore
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


def generate_dir_enc_fn(deg_view):
    """Generate directional encoding (DE) function.

    Args:
        deg_view: number of spherical harmonics degrees to use.

    Returns:
        A function for evaluating directional encoding.
    """
    integrated_dir_enc_fn = generate_ide_fn(deg_view)

    def dir_enc_fn(xyz):
        """Function returning directional encoding (DE)."""
        return integrated_dir_enc_fn(xyz, torch.zeros_like(xyz[..., :1]))

    return dir_enc_fn


#########################################################

@dataclass
class RefNerfModelConfig(ModelConfig):

    _target: Type = field(default_factory=lambda: RefNerfModel)
    num_coarse_samples: int = 64
    """Number of samples in coarse field evaluation"""
    num_importance_samples: int = 128
    """Number of samples in fine field evaluation"""

    enable_temporal_distortion: bool = False
    """Specifies whether or not to include ray warping based on time."""
    temporal_distortion_params: Dict[str, Any] = to_immutable_dict({"kind": TemporalDistortionKind.DNERF})
    """Parameters to instantiate temporal distortion with"""

    """Dimension of the appearance embedding."""
    degree_view: int = 5
    """Whether to predict normals or not."""
    predict_normals: bool = True
    """Orientation loss multiplier on computed normals."""
    orientation_loss_mult: float = 0.0001

    '''
    ### COPIED FROM REFNERFACTO
    num_proposal_samples_per_ray: Tuple[int, ...] = (256, 96)
    """Number of samples per ray for each proposal network."""
    num_nerf_samples_per_ray: int = 48
    """Number of samples per ray for the nerf network."""
    num_proposal_iterations: int = 2
    """Number of proposal network iterations."""
    use_single_jitter: bool = True
    """Whether use single jitter or not for the proposal networks."""
    proposal_net_args_list: List[Dict] = field(
        default_factory=lambda: [
            {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 128, "use_linear": False},
            {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 256, "use_linear": False},
        ]
    )
    disable_scene_contraction: bool = False
    """Whether to disable scene contraction or not."""
    use_same_proposal_network: bool = False
    """Use the same proposal network. Otherwise use different ones."""
    proposal_warmup: int = 5000
    """Scales n from 1 to proposal_update_every over this many steps"""
    proposal_update_every: int = 5
    """Sample every n steps after the warmup"""
    use_proposal_weight_anneal: bool = True
    """Whether to use proposal weight annealing."""
    proposal_weights_anneal_max_num_iters: int = 1000
    """Max num iterations for the annealing function."""
    proposal_weights_anneal_slope: float = 10.0
    """Slope of the annealing function for the proposal weights."""
    implementation: Literal["tcnn", "torch"] = "tcnn"

    proposal_initial_sampler: Literal["piecewise", "uniform"] = "piecewise"
    '''
class RefNerfModel(Model):
    """ref-NeRF model

    Args:
        config: RefNerf configuration to instantiate model
    """

    config: RefNerfModelConfig

    def __init__(
        self,
        config: RefNerfModelConfig,
        **kwargs,
    ) -> None:
        self.field = None
        assert config.collider_params is not None, "RefNerf model requires bounding box collider parameters."
        super().__init__(config=config, **kwargs)
        assert self.config.collider_params is not None, "ref-NeRF requires collider parameters to be set."

    def populate_modules(self):
        """Set the fields and modules"""
        super().populate_modules()

        # setting up fields
        position_encoding = NeRFEncoding(
            in_dim=3, num_frequencies=16, min_freq_exp=0.0, max_freq_exp=16.0, include_input=True
        )
        direction_encoding = NeRFEncoding(
            in_dim=3, num_frequencies=4, min_freq_exp=0.0, max_freq_exp=4.0, include_input=True
        )

        # Add directional encoding to the model
        if self.config.predict_normals:
            self.directional_encoding_fn = generate_dir_enc_fn(self.config.degree_view)

        self.field = RefNeRFField(aabb=self.scene_box.aabb,position_encoding=position_encoding, direction_encoding=direction_encoding, use_integrated_encoding=True)

        # samplers
        self.sampler_uniform = UniformSampler(num_samples=self.config.num_coarse_samples)
        self.sampler_pdf = PDFSampler(num_samples=self.config.num_importance_samples, include_original=False)

        # renderers
        self.renderer_rgb = RGBRenderer(background_color=colors.WHITE)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer()

        # losses
        self.rgb_loss = MSELoss()

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

        # (UNIQUE REFNERF)
        self.renderer_normals = NormalsRenderer()
        self.normals_shader = NormalsShader()


    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        if self.field is None:
            raise ValueError("populate_fields() must be called before get_param_groups")
        param_groups["fields"] = list(self.field.parameters())
        return param_groups

    def get_outputs(self, ray_bundle: RayBundle):
        if self.field is None:
            raise ValueError("populate_fields() must be called before get_outputs")

        # uniform sampling
        ray_samples_uniform = self.sampler_uniform(ray_bundle)

        # First pass:
        field_outputs_coarse = self.field.forward(ray_samples_uniform)
        weights_coarse = ray_samples_uniform.get_weights(field_outputs_coarse[FieldHeadNames.DENSITY])
        rgb_coarse = self.renderer_rgb(
            rgb=field_outputs_coarse[FieldHeadNames.RGB],
            weights=weights_coarse,
        )
        accumulation_coarse = self.renderer_accumulation(weights_coarse)
        depth_coarse = self.renderer_depth(weights_coarse, ray_samples_uniform)

        # pdf sampling
        ray_samples_pdf = self.sampler_pdf(ray_bundle, ray_samples_uniform, weights_coarse)
        #ray_samples_pdf, weights_list, ray_samples_list = self.proposal_sampler.generate_ray_samples(ray_bundle, density_fns=self.density_fns)

        # Second pass:
        field_outputs_fine = self.field.forward(ray_samples_pdf, compute_normals=self.config.predict_normals)
        weights_fine = ray_samples_pdf.get_weights(field_outputs_fine[FieldHeadNames.DENSITY])
        rgb_fine = self.renderer_rgb(
            rgb=field_outputs_fine[FieldHeadNames.RGB],
            weights=weights_fine,
        )
        accumulation_fine = self.renderer_accumulation(weights_fine)
        depth_fine = self.renderer_depth(weights_fine, ray_samples_pdf)

        # (UNIQUE REFNERF) Add directional encoding to the input features
        if self.config.predict_normals:
            directions = ray_samples_pdf.frustums.directions  # old # ray_bundle.directions
            # directions = ray_samples.metadata['directions_norm'] #not working, wrong mat sizes
            
            view_direction_features = self.directional_encoding_fn(directions)
            # repeated_view_direction_features = view_direction_features.unsqueeze(1).repeat(1, 48, 1)

            field_outputs_fine[FieldHeadNames.RGB] = torch.cat(
                [field_outputs_fine[FieldHeadNames.RGB], view_direction_features], dim=-1
            )
            # defining normals output fields
            normals = self.renderer_normals(normals=field_outputs_fine[FieldHeadNames.NORMALS], weights=weights_fine)
            pred_normals = self.renderer_normals(field_outputs_fine[FieldHeadNames.PRED_NORMALS], weights=weights_fine)
            outputs["normals"] = self.normals_shader(normals)
            outputs["pred_normals"] = self.normals_shader(pred_normals)

        outputs = {
            "rgb": rgb_coarse,
            "rgb_coarse": rgb_coarse,
            "rgb_fine": rgb_fine,
            "accumulation_coarse": accumulation_coarse,
            "accumulation_fine": accumulation_fine,
            "depth_coarse": depth_coarse,
            "depth_fine": depth_fine,
        }
        return outputs

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        image = batch["image"].to(self.device)
        pred_coarse, image_coarse = self.renderer_rgb.blend_background_for_loss_computation(
            pred_image=outputs["rgb_coarse"],
            pred_accumulation=outputs["accumulation_coarse"],
            gt_image=image,
        )
        pred_fine, image_fine = self.renderer_rgb.blend_background_for_loss_computation(
            pred_image=outputs["rgb_fine"],
            pred_accumulation=outputs["accumulation_fine"],
            gt_image=image,
        )
        rgb_loss_coarse = self.rgb_loss(image_coarse, pred_coarse)
        rgb_loss_fine = self.rgb_loss(image_fine, pred_fine)
        loss_dict = {"rgb_loss_coarse": rgb_loss_coarse, "rgb_loss_fine": rgb_loss_fine}
        loss_dict = misc.scale_dict(loss_dict, self.config.loss_coefficients)
        if self.config.predict_normals:
            # orientation loss for computed normals
            loss_dict["orientation_loss"] = self.config.orientation_loss_mult * torch.mean(
                outputs["rendered_orientation_loss"]
            )

            # ground truth supervision for normals
            loss_dict["pred_normal_loss"] = self.config.pred_normal_loss_mult * torch.mean(
                outputs["rendered_pred_normal_loss"]
            )

            # (UNIQUE REFNERF) Regularization loss for normals
            normals = outputs["normals"]  # Extract the rendered normals from the outputs
            loss_dict["normals_regularization_loss"] = self.config.orientation_loss_mult * orientation_loss(
                outputs["weights_list"], normals, outputs["ray_samples_list"]
            )
        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        assert self.config.collider_params is not None, "mip-NeRF requires collider parameters to be set."
        image = batch["image"].to(outputs["rgb_coarse"].device)
        image = self.renderer_rgb.blend_background(image)
        rgb_coarse = outputs["rgb_coarse"]
        rgb_fine = outputs["rgb_fine"]
        acc_coarse = colormaps.apply_colormap(outputs["accumulation_coarse"])
        acc_fine = colormaps.apply_colormap(outputs["accumulation_fine"])

        assert self.config.collider_params is not None
        depth_coarse = colormaps.apply_depth_colormap(
            outputs["depth_coarse"],
            accumulation=outputs["accumulation_coarse"],
            near_plane=self.config.collider_params["near_plane"],
            far_plane=self.config.collider_params["far_plane"],
        )
        depth_fine = colormaps.apply_depth_colormap(
            outputs["depth_fine"],
            accumulation=outputs["accumulation_fine"],
            near_plane=self.config.collider_params["near_plane"],
            far_plane=self.config.collider_params["far_plane"],
        )

        combined_rgb = torch.cat([image, rgb_coarse, rgb_fine], dim=1)
        combined_acc = torch.cat([acc_coarse, acc_fine], dim=1)
        combined_depth = torch.cat([depth_coarse, depth_fine], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb_coarse = torch.moveaxis(rgb_coarse, -1, 0)[None, ...]
        rgb_fine = torch.moveaxis(rgb_fine, -1, 0)[None, ...]
        rgb_coarse = torch.clip(rgb_coarse, min=0, max=1)
        rgb_fine = torch.clip(rgb_fine, min=0, max=1)

        coarse_psnr = self.psnr(image, rgb_coarse)
        fine_psnr = self.psnr(image, rgb_fine)
        fine_ssim = self.ssim(image, rgb_fine)
        fine_lpips = self.lpips(image, rgb_fine)

        assert isinstance(fine_ssim, torch.Tensor)
        metrics_dict = {
            "lpips": float(fine_lpips.item()),
            "psnr": float(fine_psnr.item()),
            "ssim": float(fine_ssim.item()),
            "coarse_psnr": float(coarse_psnr.item()),
            "fine_psnr": float(fine_psnr.item()),
            "fine_ssim": float(fine_ssim.item()),
            "fine_lpips": float(fine_lpips.item()),
        }
        images_dict = {"img": combined_rgb, "accumulation": combined_acc, "depth": combined_depth, 'original': batch["image"], 'output': rgb_fine}
        return metrics_dict, images_dict


    '''
    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        if self.config.disable_scene_contraction:
            scene_contraction = None
        else:
            scene_contraction = SceneContraction(order=float("inf"))

        # Add directional encoding to the model
        if self.config.predict_normals:
            self.directional_encoding_fn = generate_dir_enc_fn(self.config.degree_view)

        # Fields
        self.field = NerfactoField(
            self.scene_box.aabb,
            spatial_distortion=scene_contraction,
            num_images=self.num_train_data,
        )

        self.density_fns = []
        num_prop_nets = self.config.num_proposal_iterations
        # Build the proposal network(s)
        self.proposal_networks = torch.nn.ModuleList()
        if self.config.use_same_proposal_network:
            assert len(self.config.proposal_net_args_list) == 1, "Only one proposal network is allowed."
            prop_net_args = self.config.proposal_net_args_list[0]
            network = HashMLPDensityField(
                self.scene_box.aabb,
                spatial_distortion=scene_contraction,
                **prop_net_args,
                implementation=self.config.implementation,
            )
            self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for _ in range(num_prop_nets)])
        else:
            for i in range(num_prop_nets):
                prop_net_args = self.config.proposal_net_args_list[min(i, len(self.config.proposal_net_args_list) - 1)]
                network = HashMLPDensityField(
                    self.scene_box.aabb,
                    spatial_distortion=scene_contraction,
                    **prop_net_args,
                    implementation=self.config.implementation,
                )
                self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for network in self.proposal_networks])

        # Samplers
        def update_schedule(step):
            return np.clip(
                np.interp(step, [0, self.config.proposal_warmup], [0, self.config.proposal_update_every]),
                1,
                self.config.proposal_update_every,
            )

        # Change proposal network initial sampler if uniform
        initial_sampler = None  # None is for piecewise as default (see ProposalNetworkSampler)
        if self.config.proposal_initial_sampler == "uniform":
            initial_sampler = UniformSampler(single_jitter=self.config.use_single_jitter)

        self.proposal_sampler = ProposalNetworkSampler(
            num_nerf_samples_per_ray=self.config.num_nerf_samples_per_ray,
            num_proposal_samples_per_ray=self.config.num_proposal_samples_per_ray,
            num_proposal_network_iterations=self.config.num_proposal_iterations,
            single_jitter=self.config.use_single_jitter,
            update_sched=update_schedule,
            initial_sampler=initial_sampler,
        )


    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        param_groups["proposal_networks"] = list(self.proposal_networks.parameters())
        param_groups["fields"] = list(self.field.parameters())
        return param_groups

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = []
        if self.config.use_proposal_weight_anneal:
            # anneal the weights of the proposal network before doing PDF sampling
            N = self.config.proposal_weights_anneal_max_num_iters

            def set_anneal(step):
                # https://arxiv.org/pdf/2111.12077.pdf eq. 18
                train_frac = np.clip(step / N, 0, 1)

                def bias(x, b):
                    return b * x / ((b - 1) * x + 1)

                anneal = bias(train_frac, self.config.proposal_weights_anneal_slope)
                self.proposal_sampler.set_anneal(anneal)

            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=set_anneal,
                )
            )
            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=self.proposal_sampler.step_cb,
                )
            )
        return callbacks

    '''
