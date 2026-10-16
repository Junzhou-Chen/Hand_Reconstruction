#!/usr/bin/env python3
"""Run RGB -> official HaWoR -> feedback, given detections/camera metadata.

The input track JSON has frames [{image,bbox_xyxy,is_right,intrinsics,
camera_to_world,timestamp,detected,bbox_valid?,image_valid?,reset?}]. Image paths
are relative to the JSON file. Camera poses must be supplied by the frontend;
this command does not silently replace SLAM by identity or estimate a detector.
"""
from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path

import torch

from handloop.models.visual_feedback import FeedbackHistory, VisualFeedbackConfig, VisualFeedbackSystem
from handloop.models.probabilistic_twm import ProbabilisticTWM, ProbabilisticTWMConfig
from handloop.visual.hawor import FrozenHaWoRBridge
from handloop.visual.pipeline import HandLoopFeedbackPipeline
from train_visual_feedback import load_feedback_system, atomic_json, utc
from handloop.data.visual_clips import file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--track-json', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--hawor-root', type=Path, required=True)
    parser.add_argument('--hawor-checkpoint', type=Path, required=True)
    parser.add_argument('--hawor-config', type=Path, required=True)
    parser.add_argument('--mano-models', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--history', type=Path)
    parser.add_argument('--forecast', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a new prediction output file')
    document = json.loads(args.track_json.read_text())
    frames = document['frames']
    if not frames or document.get('translation_units') != 'metres':
        raise ValueError('A nonempty track with explicit translation_units=metres is required')
    if not document.get('coordinate_frame') or not document.get('timestamp_source'):
        raise ValueError('Record coordinate frame and timestamp provenance')
    device = torch.device(args.device)
    torch.set_num_threads(1)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    joint = 'replay_config' in checkpoint and 'identity' in checkpoint
    if joint:
        twm = ProbabilisticTWM(ProbabilisticTWMConfig(**checkpoint['twm_source']['model_config']))
        feedback = VisualFeedbackSystem(twm, VisualFeedbackConfig(**checkpoint['config']['adapter']))
        for side in ('LEFT', 'RIGHT'):
            if file_sha256(args.mano_models/f'MANO_{side}.pkl') != checkpoint['identity']['mano_asset_sha256'][side]:
                raise ValueError('Joint checkpoint MANO assets differ')
        if file_sha256(args.hawor_config) != checkpoint['identity']['hawor_config_sha256']:
            raise ValueError('Joint checkpoint HaWoR architecture config differs')
    else:
        del checkpoint
        feedback, checkpoint = load_feedback_system(args.checkpoint, device)
    frontend = FrozenHaWoRBridge.from_pretrained(args.hawor_root, args.hawor_checkpoint,
        args.hawor_config, args.mano_models, device=str(device))
    pipeline = HandLoopFeedbackPipeline(frontend, feedback).eval()
    if joint:
        prefix = 'pipeline.'
        if not all(name.startswith(prefix) for name in checkpoint['model']):
            raise ValueError('Unexpected joint checkpoint parameter namespace')
        pipeline.load_state_dict({name[len(prefix):]: value for name, value in checkpoint['model'].items()}, strict=True)
        pipeline.to(device).eval()
    # Do not retain the large optimizer and CPU state during RGB inference.
    checkpoint_step = checkpoint['step']
    del checkpoint
    initial = None
    if args.history:
        state = torch.load(args.history, map_location=device, weights_only=False)
        initial = FeedbackHistory(**state)
    def values(name, default=None):
        return [frame[name] if default is None else frame.get(name, default) for frame in frames]
    with torch.no_grad():
        output = pipeline.run_track(
            [(args.track_json.parent / f['image']).resolve() for f in frames],
            values('bbox_xyxy'), values('is_right'), values('intrinsics'), values('camera_to_world'),
            values('timestamp'), detected=values('detected'), bbox_valid=values('bbox_valid', True),
            image_valid=values('image_valid', True), reset=values('reset', False),
            initial_history=initial, return_forecast=args.forecast)
    # Export only numerical predictions/distribution descriptors, not image features.
    keys = ('state_world', 'candidate_world', 'raw_world', 'state_translation_std',
            'state_translation_covariance_world', 'state_rotation_concentration', 'anchor_world',
            'availability', 'candidate_availability', 'frame_valid', 'image_valid',
            'history_count', 'history_reset', 'detected', 'bbox_valid', 'forecast_world',
            'forecast_modes_world', 'mixture_weights', 'forecast_translation_std',
            'forecast_modes_translation_covariance_world', 'forecast_rotation_concentration',
            'beta', 'refinement_states_world', 'refinement_candidates_world',
            'refinement_joint_crop_coords', 'refinement_joint_projection_valid')
    result = {key: output[key].detach().cpu() for key in keys if key in output}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + '.tmp')
    torch.save(result, temporary); temporary.replace(args.output)
    history_path = args.output.with_suffix('.history.pt')
    torch.save({field.name: getattr(output['final_history'], field.name).detach().cpu()
                for field in fields(FeedbackHistory)}, history_path)
    atomic_json(args.output.with_suffix('.json'), {
        'utc': utc(), 'frames': len(frames), 'output': str(args.output),
        'history': str(history_path), 'feedback_checkpoint_sha256': file_sha256(args.checkpoint),
        'feedback_step': checkpoint_step, 'joint_checkpoint': joint, 'frontend': frontend.provenance(),
        'track_json_sha256': file_sha256(args.track_json),
        'coordinate_frame': document['coordinate_frame'],
        'timestamp_source': document['timestamp_source'],
        'input_protocol': document.get('input_protocol', 'external supplied detections and cameras'),
        'translation_units': 'metres', 'rotation_encoding': 'first two columns; c0xyz+c1xyz',
        'uncertainty': 'std and NLL in per-frame anchor; full covariance also exported in world axes',
        'lookahead': 'original nonoverlapping 16-frame HaWoR blocks; not strictly online',
        'scope': 'precomputed detections/camera input, no motion infiller; no accuracy claim'})
    print(json.dumps({'frames': len(frames), 'output': str(args.output), 'history': str(history_path)}))


if __name__ == '__main__':
    main()
