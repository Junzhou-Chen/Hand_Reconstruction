"""RGB crops -> frozen official HaWoR -> feedback -> final state distribution.

Bounding boxes and camera poses are explicit frontend inputs. The original
HaWoR detector/SLAM can supply them; this module does not invent camera poses
or call GT calibration an estimated egocentric world trajectory.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

from handloop.models.visual_feedback import FeedbackHistory, VisualFeedbackSystem, inverse_rigid
from .hawor import FrozenHaWoRBridge


def effective_crop_transform(full_crop_transform: Tensor) -> Tensor:
    """Source pixels -> [-1,1] in HaWoR's central 192 x 256 image region."""
    # Pixel centers: u_norm=2*(u_crop-32+.5)/192-1; likewise v.
    affine = full_crop_transform.new_tensor([[2 / 192, 0, -1 - 63 / 192],
                                           [0, 2 / 256, -1 + 1 / 256], [0, 0, 1]])
    return affine @ full_crop_transform


class HandLoopFeedbackPipeline(nn.Module):
    """Frozen RGB feature path plus a trainable visual feedback readout.

    ``forward`` expects prepared 16-frame crops and real camera/crop metadata.
    ``run_track`` reads RGB files using the official crop preprocessing. Both
    retain original bidirectional HaWoR context; neither implies online SLAM.
    A completed ``final_history`` can be passed to the next block/track call.
    Track identity changes must explicitly reset history.
    """

    def __init__(self, frontend: FrozenHaWoRBridge, feedback: VisualFeedbackSystem):
        super().__init__()
        self.frontend = frontend
        self.feedback = feedback

    def train(self, mode: bool = True):
        super().train(mode)
        self.frontend.eval()
        return self

    def forward(self, batch: dict[str, Tensor], *, initial_history: Optional[FeedbackHistory] = None,
                return_forecast: bool = False, use_adapter: bool = True,
                adapter_history: bool = True) -> dict:
        """No GT arguments: targets belong exclusively to the external loss.

        Extra metadata: timestamps[B,16], intrinsics[B,16,3,3],
        crop_transform[B,16,3,3] already normalized to the effective crop,
        optional reset[B,16]. Other batch entries follow FrozenHaWoRBridge.
        """
        observed = self.frontend(batch)
        output = self.feedback.forward_clip(
            observed['image_features'], observed['raw_state_world'],
            timestamps=batch['timestamps'], frame_valid=observed['frame_valid'],
            availability=observed['raw_available'],
            image_valid=observed['image_valid'] & observed['bbox_valid'],
            handedness=observed['is_right'].long(),
            world_to_camera=inverse_rigid(observed['camera_to_world']),
            camera_intrinsics=batch['intrinsics'], crop_transform=batch['crop_transform'],
            beta=observed['beta'], reset=batch.get('reset'), initial_history=initial_history,
            return_forecast=return_forecast, use_adapter=use_adapter, adapter_history=adapter_history)
        output['detected'] = observed['detected']
        output['bbox_valid'] = observed['bbox_valid']
        return output

    @torch.no_grad()
    def run_track(self, image_paths, bbox_xyxy, is_right, intrinsics, camera_to_world,
                  timestamps, *, detected=None, bbox_valid=None, frame_valid=None,
                  image_valid=None, reset=None, initial_history=None,
                  return_forecast=False, use_adapter=True, adapter_history=True) -> dict:
        """Inference on one chronological RGB track, preserving block feedback.

        Inputs are camera estimates and detections, not supervision. Empty or
        interpolated detections retain their actual observation masks. Final
        short HaWoR chunks are padded inside the frozen frontend only.
        """
        self.eval()
        if detected is None:
            raise ValueError('Provide the original detection mask separately from interpolated boxes')
        observed = self.frontend.extract_clip(
            image_paths, bbox_xyxy, is_right, intrinsics, camera_to_world,
            detected=detected, bbox_valid=bbox_valid,
            image_valid=image_valid)
        device = next(self.feedback.parameters()).device
        observed = {key: value.to(device) if isinstance(value, Tensor) else value
                    for key, value in observed.items()}
        n = len(image_paths)
        if frame_valid is not None:
            valid = torch.as_tensor(frame_valid, dtype=torch.bool, device=device)
            if valid.shape != (n,):
                raise ValueError('frame_valid must have one value per RGB frame')
            observed['frame_valid'] &= valid
            observed['raw_available'] &= valid[:, None]
            observed['image_valid'] &= valid
        time = torch.as_tensor(timestamps, dtype=torch.float64, device=device)
        if time.shape != (n,):
            raise ValueError('timestamps must be one value per input RGB frame')
        output = self.feedback.forward_clip(
            observed['image_features'][None], observed['raw_state_world'][None],
            timestamps=time[None], frame_valid=observed['frame_valid'][None],
            availability=observed['raw_available'][None],
            image_valid=(observed['image_valid'] & observed['bbox_valid'])[None],
            handedness=observed['is_right'].long()[None],
            world_to_camera=inverse_rigid(observed['camera_to_world'])[None],
            camera_intrinsics=observed['intrinsics'][None],
            crop_transform=effective_crop_transform(observed['crop_transform'])[None],
            beta=observed['beta'][None],
            reset=None if reset is None else torch.as_tensor(reset, device=device, dtype=torch.bool)[None],
            initial_history=initial_history, return_forecast=return_forecast,
            use_adapter=use_adapter, adapter_history=adapter_history)
        output['detected'] = observed['detected'][None]
        output['bbox_valid'] = observed['bbox_valid'][None]
        return output
