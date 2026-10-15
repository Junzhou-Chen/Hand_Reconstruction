"""Deterministic finite-coverage epochs, disjoint DDP shards and batch resume.

For even N/world, every source window appears exactly once per full epoch.
For odd N, DistributedSampler pads at most world-1 indices, explicitly logged.
The batch cursor is stored in optimizer units and applied before dataset IO.
"""
from torch.utils.data import DistributedSampler


class CoverageEpochSampler(DistributedSampler):
    def __init__(self, dataset, *, replicas, rank, seed, batch_size):
        super().__init__(dataset,num_replicas=replicas,rank=rank,shuffle=True,seed=seed,drop_last=False)
        self.batch_size=batch_size
        self.batch_cursor=0

    def __iter__(self):
        return iter(list(super().__iter__())[self.batch_cursor*self.batch_size:])

    def state_dict(self):
        return {'epoch':self.epoch,'batch_cursor':self.batch_cursor,'seed':self.seed,
                'num_samples':self.num_samples,'batch_size':self.batch_size,'rank':self.rank,
                'replicas':self.num_replicas}

    def load_state_dict(self,state):
        expected=self.state_dict()
        for key in ('seed','num_samples','batch_size','rank','replicas'):
            if state[key]!=expected[key]: raise ValueError(f'Epoch sampler mismatch: {key}')
        self.set_epoch(state['epoch']); self.batch_cursor=state['batch_cursor']
