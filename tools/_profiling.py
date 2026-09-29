import json
import os
import statistics
import time

import torch


def add_profile_args(parser):
    parser.add_argument(
        '--profile',
        action='store_true',
        help='profile forward-only inference time and peak GPU memory',
    )
    parser.add_argument(
        '--profile-warmup',
        type=int,
        default=10,
        help='number of warmup scenes excluded from profiling statistics',
    )
    parser.add_argument(
        '--profile-max-scenes',
        type=int,
        default=0,
        help='number of scenes to profile after warmup; 0 means profile the full set',
    )
    parser.add_argument(
        '--profile-json',
        type=str,
        help='optional path to save profiling summary as JSON',
    )
    parser.add_argument(
        '--skip-eval',
        action='store_true',
        help='skip quantitative evaluation after inference',
    )
    return parser


def parameter_count_million(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


class InferenceProfiler:
    def __init__(self, enabled=False, warmup_scenes=10, max_profile_scenes=0, logger=None):
        self.enabled = enabled
        self.warmup_scenes = max(int(warmup_scenes), 0)
        self.max_profile_scenes = max(int(max_profile_scenes), 0)
        self.logger = logger

        self.num_seen = 0
        self.times_ms = []
        self.peak_allocated_gb = []
        self.peak_reserved_gb = []

    def before_forward(self):
        if not self.enabled:
            return None
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        return time.perf_counter()

    def after_forward(self, start_time, batch_size=1):
        if not self.enabled or start_time is None:
            return

        batch_size = max(int(batch_size), 1)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0 / batch_size

        peak_allocated_gb = 0.0
        peak_reserved_gb = 0.0
        if torch.cuda.is_available():
            peak_allocated_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
            peak_reserved_gb = torch.cuda.max_memory_reserved() / (1024 ** 3)

        self.num_seen += 1
        is_warmup = self.num_seen <= self.warmup_scenes
        should_record = (not is_warmup) and (
            self.max_profile_scenes == 0 or len(self.times_ms) < self.max_profile_scenes
        )
        if should_record:
            self.times_ms.append(elapsed_ms)
            self.peak_allocated_gb.append(peak_allocated_gb)
            self.peak_reserved_gb.append(peak_reserved_gb)

    def should_stop(self):
        return self.enabled and self.max_profile_scenes > 0 and len(self.times_ms) >= self.max_profile_scenes

    def summary(self, params_m, model_name=None):
        summary = {
            'model_name': model_name,
            'params_m': round(float(params_m), 4),
            'warmup_scenes': self.warmup_scenes,
            'profiled_scenes': len(self.times_ms),
            'measurement': 'forward_only_excludes_dataloader_and_evaluation',
        }
        if torch.cuda.is_available():
            summary['device_name'] = torch.cuda.get_device_name(torch.cuda.current_device())

        if self.times_ms:
            summary.update(
                {
                    'avg_time_ms_per_scene': round(float(statistics.mean(self.times_ms)), 4),
                    'median_time_ms_per_scene': round(float(statistics.median(self.times_ms)), 4),
                    'std_time_ms_per_scene': round(
                        float(statistics.pstdev(self.times_ms)) if len(self.times_ms) > 1 else 0.0,
                        4,
                    ),
                    'avg_peak_allocated_gb': round(float(statistics.mean(self.peak_allocated_gb)), 4),
                    'max_peak_allocated_gb': round(float(max(self.peak_allocated_gb)), 4),
                    'avg_peak_reserved_gb': round(float(statistics.mean(self.peak_reserved_gb)), 4),
                    'max_peak_reserved_gb': round(float(max(self.peak_reserved_gb)), 4),
                }
            )
        return summary

    def log_summary(self, params_m, model_name=None):
        summary = self.summary(params_m=params_m, model_name=model_name)
        if self.logger is not None:
            self.logger.info('PROFILE_RESULT ' + json.dumps(summary, sort_keys=True))
        return summary


def save_profile_json(path, summary):
    if not path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, sort_keys=True)
