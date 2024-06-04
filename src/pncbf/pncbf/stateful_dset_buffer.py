import numpy as np

from pncbf.pncbf.pncbf import PNCBF
from pncbf.utils.jax_utils import jax2np, tree_cat


class StatefulDsetBuffer:
    def __init__(self, seed: int, dset_len_max: int = 8):
        self._rng = np.random.default_rng(seed=seed)
        self._dset_len_max = dset_len_max
        self._dset_list = []

        self._dset: PNCBF.CollectData | None = None

    @property
    def dset_len_max(self):
        return self._dset_len_max

    @property
    def b(self):
        return self._dset.bT_x.shape[0]

    @property
    def is_full(self) -> bool:
        return len(self._dset_list) >= self.dset_len_max

    @property
    def b_times_Tm1(self):
        return self.b * (self._dset.bT_x.shape[1] - 1)

    def add_dset(self, dset: PNCBF.CollectData, get_vterms_fn):
        self._dset_list.append(jax2np(dset))

        if len(self._dset_list) > self._dset_len_max:
            del self._dset_list[0]

        for ii, dset_item in enumerate(self._dset_list[:-1]):
            self._dset_list[ii] = self._dset_list[ii]._replace(b_vterms=jax2np(get_vterms_fn(dset_item.bT_x)))

        self._dset = tree_cat(self._dset_list, axis=0)

    def sample_batch(self, n_rng: int, n_zero: int) -> PNCBF.Batch:
        b_idx_rng = self._rng.integers(0, self.b_times_Tm1, size=(n_rng,))
        b_idx_b_rng = b_idx_rng // (self._dset.bT_x.shape[1] - 1)
        b_idx_t_rng = 1 + (b_idx_rng % (self._dset.bT_x.shape[1] - 1))

        b_idx_b_zero = self._rng.integers(0, self.b, size=(n_zero,))
        b_idx_t_zero = np.zeros_like(b_idx_b_zero)

        b_idx_b = np.concatenate([b_idx_b_rng, b_idx_b_zero], axis=0)
        b_idx_t = np.concatenate([b_idx_t_rng, b_idx_t_zero], axis=0)

        b_x0 = self._dset.bT_x[b_idx_b, b_idx_t]
        b_u0 = self._dset.bT_u[b_idx_b, b_idx_t]
        b_xT = self._dset.bT_x[b_idx_b, -1]
        bh_lhs = self._dset.b_vterms.Th_max_lhs[b_idx_b, b_idx_t, :]
        bh_int_rhs = self._dset.b_vterms.Th_disc_int_rhs[b_idx_b, b_idx_t, :]
        b_discount_rhs = self._dset.b_vterms.T_discount_rhs[b_idx_b, b_idx_t]

        bh_iseqh = None
        if self._dset.bTh_iseqh is not None:
            bh_iseqh = self._dset.bTh_iseqh[b_idx_b, b_idx_t, :]

        batch = PNCBF.Batch(b_x0, b_u0, b_xT, bh_iseqh, bh_lhs, bh_int_rhs, b_discount_rhs)

        return batch
