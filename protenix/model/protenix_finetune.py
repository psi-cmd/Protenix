import copy
import random
import time
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn

from protenix.model import sample_confidence
from protenix.model.generator import (
    InferenceNoiseScheduler,
    TrainingNoiseSampler,
    sample_diffusion,
    sample_diffusion_training,
)
from protenix.model.utils import simple_merge_dict_list
from protenix.openfold_local.model.primitives import LayerNorm
from protenix.utils.logger import get_logger
from protenix.utils.permutation.permutation import SymmetricPermutation
from protenix.utils.torch_utils import autocasting_disable_decorator
from protenix.utils.torch_utils import to_device

from .modules.confidence import ConfidenceHead
from .modules.diffusion import DiffusionModule
from .modules.finetune_blocks import FinetuneBlock
from .modules.embedders import InputFeatureEmbedder, RelativePositionEncoding
from .modules.head import DistogramHead
from .modules.pairformer import MSAModule, PairformerStack, TemplateEmbedder
from .modules.primitives import LinearNoBias
from .protenix import Protenix


class ProtenixFinetune(Protenix):
    """
    ProtenixFinetune is a subclass of Protenix.
    It is used to finetune the Protenix model on the molecular glue dataset.

    Finetune Plan:
    1. Extract and embed molecular glue environment with e3nn
    2. Cross-attention the molecular glue environment with the protein structure (1d info first)
    3. Pass as input to the diffusion process
    4. Feedback the environment during the diffusion process back to the input of e3nn

    Details:
    1. There is a parameter controlling the feedback strength during the training (teacher-forcing)
    2. After training, the model should direct itself purely from the diffusion process.
    3. Train diffusion module only. (without pairformer and confidence head)

    """
    def __init__(self, configs):
        super().__init__(configs)
        # Freeze all other parameters
        for param in self.parameters():
            param.requires_grad = False

        self.finetune_block = FinetuneBlock(configs)

    def get_input_for_e3nn(self, input_feature_dict, label_full_dict, label_dict):
        """
        Input molecular glue environment:
        - molecular glue is selected by input_feature_dict["is_glue"]
        - select atoms within 5A of the molecular glue in input_feature_dict["ref_pos"]
        - get directional input for e3nn

        """
        pass


    def forward(self, input_feature_dict, label_full_dict, label_dict, mode="train", current_step=None, symmetric_permutation=None):
        """
        Forward pass of the Alphafold3 model.

        Args:
            input_feature_dict (dict[str, Any]): Input features dictionary.
            label_full_dict (dict[str, Any]): Full label dictionary (uncropped).
            label_dict (dict[str, Any]): Label dictionary (cropped).
            mode (str): Mode of operation ('train', 'inference', 'eval'). Defaults to 'inference'.
            current_step (Optional[int]): Current training step. Defaults to None.
            symmetric_permutation (SymmetricPermutation): Symmetric permutation object. Defaults to None.

        Returns:
            tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
                Prediction, updated label, and log dictionaries.
        """

        assert mode in ["train", "inference", "eval"]
        inplace_safe = not (self.training or torch.is_grad_enabled())
        chunk_size = self.configs.infer_setting.chunk_size if inplace_safe else None

        if mode == "train":
            nc_rng = np.random.RandomState(current_step)
            N_cycle = nc_rng.randint(1, self.N_cycle + 1)
            assert self.training
            assert label_dict is not None
            assert symmetric_permutation is not None

            pred_dict, label_dict, log_dict = self.main_train_loop(
                input_feature_dict=input_feature_dict,
                label_full_dict=label_full_dict,
                label_dict=label_dict,
                N_cycle=N_cycle,
                symmetric_permutation=symmetric_permutation,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
            )
        elif mode == "inference":
            pred_dict, log_dict, time_tracker = self.main_inference_loop(
                input_feature_dict=input_feature_dict,
                label_dict=None,
                N_cycle=self.N_cycle,
                mode=mode,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
                N_model_seed=self.N_model_seed,
                symmetric_permutation=None,
            )
            log_dict.update({"time": time_tracker})
        elif mode == "eval":
            if label_dict is not None:
                assert (
                    label_dict["coordinate"].size()
                    == label_full_dict["coordinate"].size()
                )
                label_dict.update(label_full_dict)

            pred_dict, log_dict, time_tracker = self.main_inference_loop(
                input_feature_dict=input_feature_dict,
                label_dict=label_dict,
                N_cycle=self.N_cycle,
                mode=mode,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
                N_model_seed=self.N_model_seed,
                symmetric_permutation=symmetric_permutation,
            )
            log_dict.update({"time": time_tracker})

        return pred_dict, label_dict, log_dict
    
    def main_train_loop(
        self,
        input_feature_dict: dict[str, Any],
        label_full_dict: dict[str, Any],
        label_dict: dict,
        N_cycle: int,
        symmetric_permutation: SymmetricPermutation,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
        """
        Main training loop for the Alphafold3 model.

        Args:
            input_feature_dict (dict[str, Any]): Input features dictionary.
            label_full_dict (dict[str, Any]): Full label dictionary (uncropped).
            label_dict (dict): Label dictionary (cropped).
            N_cycle (int): Number of cycles.
            symmetric_permutation (SymmetricPermutation): Symmetric permutation object.
            inplace_safe (bool): Whether to use inplace operations safely. Defaults to False.
            chunk_size (Optional[int]): Chunk size for memory-efficient operations. Defaults to None.

        Returns:
            tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
                Prediction, updated label, and log dictionaries.
        """
        N_token = input_feature_dict["residue_index"].shape[-1]
        if N_token <= 16:
            deepspeed_evo_attention_condition_satisfy = False
        else:
            deepspeed_evo_attention_condition_satisfy = True

        # We won't train the pairformer during finetune
        with torch.no_grad():
            s_inputs, s, z = self.get_pairformer_output(
                input_feature_dict=input_feature_dict,
                N_cycle=N_cycle,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
            )

        log_dict = {}
        pred_dict = {}

        if input_feature_dict["is_glue"].sum() == 0:
            finetune_block = None
            print("No glue found, skip finetune block")
        else:
            finetune_block = self.finetune_block

        # Mini-rollout: used for confidence and label permutation
        with torch.no_grad():
            # [..., 1, N_atom, 3]
            N_sample_mini_rollout = self.configs.sample_diffusion[
                "N_sample_mini_rollout"
            ]  # =1
            N_step_mini_rollout = self.configs.sample_diffusion["N_step_mini_rollout"]

            coordinate_mini = self.sample_diffusion(
                denoise_net=self.diffusion_module,
                input_feature_dict=input_feature_dict,
                s_inputs=s_inputs.detach(),
                s_trunk=s.detach(),
                z_trunk=z.detach(),
                N_sample=N_sample_mini_rollout,
                noise_schedule=self.inference_noise_scheduler(
                    N_step=N_step_mini_rollout,
                    device=s_inputs.device,
                    dtype=s_inputs.dtype,
                ),
            )
            coordinate_mini.detach_()
            pred_dict["coordinate_mini"] = coordinate_mini

            # Permute ground truth to match mini-rollout prediction
            label_dict, perm_log_dict = (
                symmetric_permutation.permute_label_to_match_mini_rollout(
                    coordinate_mini,
                    input_feature_dict,
                    label_dict,
                    label_full_dict,
                )
            )
            log_dict.update(perm_log_dict)
        
        # No need to run confidence head during finetune
        
        # Confidence: use mini-rollout prediction, and detach token embeddings
        # drop_embedding = (
        #     random.random() < self.configs.model.confidence_embedding_drop_rate
        # )
        # plddt_pred, pae_pred, pde_pred, resolved_pred = self.run_confidence_head(
        #     input_feature_dict=input_feature_dict,
        #     s_inputs=s_inputs,
        #     s_trunk=s,
        #     z_trunk=z,
        #     pair_mask=None,
        #     x_pred_coords=coordinate_mini,
        #     use_embedding=not drop_embedding,
        #     use_memory_efficient_kernel=self.configs.use_memory_efficient_kernel,
        #     use_deepspeed_evo_attention=self.configs.use_deepspeed_evo_attention
        #     and deepspeed_evo_attention_condition_satisfy,
        #     use_lma=self.configs.use_lma,
        #     inplace_safe=inplace_safe,
        #     chunk_size=chunk_size,
        # )
        # pred_dict.update(
        #     {
        #         "plddt": plddt_pred,
        #         "pae": pae_pred,
        #         "pde": pde_pred,
        #         "resolved": resolved_pred,
        #     }
        # )

        if self.train_confidence_only:
            # Skip diffusion loss and distogram loss. Return now.
            return pred_dict, label_dict, log_dict

        # Denoising: use permuted coords to generate noisy samples and perform denoising
        # x_denoised: [..., N_sample, N_atom, 3]
        # x_noise_level: [..., N_sample]

        

        with torch.no_grad():
            N_sample = self.diffusion_batch_size
            drop_conditioning = (
                random.random() < self.configs.model.condition_embedding_drop_rate
            )
            # 旋转增强的坐标，扩散结果，噪声水平，原始坐标+噪声
            x_gt_augment, x_denoised, x_noise_level, x_noisy = autocasting_disable_decorator(
                self.configs.skip_amp.sample_diffusion_training
            )(sample_diffusion_training)(
                noise_sampler=self.train_noise_sampler,
                denoise_net=self.diffusion_module,
                label_dict=label_dict,
                input_feature_dict=input_feature_dict,
                s_inputs=s_inputs,
                s_trunk=s,
                z_trunk=z,
                N_sample=N_sample,
                diffusion_chunk_size=self.configs.diffusion_chunk_size,
                use_conditioning=not drop_conditioning,
                finetune_block=finetune_block,
            )

        
        pred_dict.update(
            {
                "distogram": autocasting_disable_decorator(True)(self.distogram_head)(
                    z
                ),
                # [..., N_sample=48, N_atom, 3]: diffusion loss
                "coordinate": x_denoised,
                "noise_level": x_noise_level,
            }
        )

        # Permute symmetric atom/chain in each sample to match true structure
        # Note: currently chains cannot be permuted since label is cropped
        pred_dict, perm_log_dict, _, _ = (
            symmetric_permutation.permute_diffusion_sample_to_match_label(
                input_feature_dict, pred_dict, label_dict, stage="train"
            )
        )
        log_dict.update(perm_log_dict)

        return pred_dict, label_dict, log_dict

        