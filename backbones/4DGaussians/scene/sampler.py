import numpy as np
from torch.utils.data.sampler import Sampler


class UniformTimeSampler(Sampler):

    def __init__(self, dataset, frame_count=300, gt_delta_time=None, seed=6666):
        self.dataset = dataset
        self.frame_count = frame_count
        self.gt_delta_time = gt_delta_time or []
        self.rng = np.random.default_rng(seed)

        self.t_interval = 2.0 / (frame_count - 1)
        self.t_grid = np.array([-1.0 + k * self.t_interval for k in range(frame_count)])
        self.t_to_samples = {t: [] for t in self.t_grid}
        self._build_mappings()

    def _build_mappings(self):
        tolerance = self.t_interval / 2.0
        for sample_idx in range(len(self.dataset)):
            cam = self.dataset[sample_idx]
            cam_time = getattr(cam, "time", None)
            cam_idx = getattr(cam, "cam_index", None)
            if cam_time is None:
                continue
            if self.gt_delta_time and cam_idx is not None and cam_idx < len(self.gt_delta_time):
                t = float(cam_time) + self.gt_delta_time[cam_idx]
            else:
                t = float(cam_time)
            closest_t = self.t_grid[np.argmin(np.abs(self.t_grid - t))]
            if abs(closest_t - t) <= tolerance:
                self.t_to_samples[closest_t].append(sample_idx)

    def coverage(self):
        return np.array([len(self.t_to_samples[t]) for t in self.t_grid], dtype=int)

    def __iter__(self):
        probs = np.ones(len(self.t_grid)) / len(self.t_grid)
        empty_draws = 0
        while True:
            t = self.rng.choice(self.t_grid, p=probs)
            samples = self.t_to_samples[t]
            if len(samples) == 0:
                empty_draws += 1
                if empty_draws >= 1000:
                    raise RuntimeError("UniformTimeSampler: no training view at the sampled time step; "
                                       "check gt_delta_time and frame_count.")
                continue
            empty_draws = 0
            yield self.rng.choice(samples)

    def __len__(self):
        return len(self.dataset)
