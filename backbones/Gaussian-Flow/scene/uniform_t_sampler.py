import numpy as np
from torch.utils.data import Sampler

__all__ = ["UniformTSampler"]


class UniformTSampler(Sampler):
    """Frame-level uniform sampling across time (Sec. 3.1) for the Gaussian-Flow backbone.

    The loader gives every training image its global frame index s_k + j (see
    scene/neural_3D_dataset_NDC.py). A time step is drawn uniformly among the steps that
    have at least one view, then one of its views uniformly, so every time step receives
    the same expected number of updates regardless of how many cameras observe it.
    Draws are with replacement and never end. Times are passed as an array so that no
    image is decoded at construction.
    """

    def __init__(self, times, frame_count=300, seed=6666):
        super().__init__(None)
        self.rng = np.random.default_rng(seed)
        t = np.asarray(times, dtype=np.float64)
        self.n_samples = int(t.size)

        self.t_grid = np.arange(frame_count, dtype=np.float64)
        g = np.abs(t[:, None] - self.t_grid[None, :]).argmin(axis=1)
        self.t_to_samples = {}
        for gi in np.unique(g):
            self.t_to_samples[int(gi)] = np.flatnonzero(g == gi)

        self._covered = np.fromiter(self.t_to_samples.keys(), dtype=np.int64)
        if self._covered.size == 0:
            raise RuntimeError("UniformTSampler: no time step has a view; check frame_count and the data")
        cnt = np.array([self.t_to_samples[int(x)].size for x in self._covered])
        print(f"UniformTSampler: {self.n_samples} views over {frame_count} time steps "
              f"({self._covered.size} steps with views, {cnt.min()}~{cnt.max()} views per step)")

    def __len__(self):
        return self.n_samples

    def __iter__(self):
        while True:
            gi = int(self.rng.choice(self._covered))
            yield int(self.rng.choice(self.t_to_samples[gi]))
