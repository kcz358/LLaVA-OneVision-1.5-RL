import os
import random
from torch.utils.data import BatchSampler, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader


class FilteredRandomSampler(Sampler):
    def __init__(self, dataset, mastered_samples, generator=None):
        self.dataset = dataset
        self.mastered_samples = mastered_samples
        self.generator = generator
        self._init_indices()

    def _init_indices(self):
        all_indices = list(range(len(self.dataset)))
        self.filtered_indices = []
        for idx in all_indices:
            sample = self.dataset[idx]
            qid = sample.get("query_id") or sample.get("id") or sample.get("qid")
            if qid not in self.mastered_samples:
                self.filtered_indices.append(idx)

    def __iter__(self):
        if self.generator is None:
            gen = random
        else:
            gen = self.generator
        indices = self.filtered_indices.copy()
        gen.shuffle(indices)
        return iter(indices)

    def __len__(self):
        return len(self.filtered_indices)

    def add_mastered_sample(self, qid):
        if qid not in self.mastered_samples:
            self.mastered_samples.add(qid)
            self._init_indices()


class FilteredStatefulDataLoader(StatefulDataLoader):
    def __init__(self, dataset, mastered_samples_file=None, *args, **kwargs):
        shuffle = kwargs.pop("shuffle")
        batch_size = kwargs.get("batch_size")
        drop_last = kwargs.get("drop_last")
        self.shuffle = shuffle
        self.mastered_samples_file = mastered_samples_file
        self.mastered_samples = set()

        if mastered_samples_file and os.path.exists(mastered_samples_file):
            with open(mastered_samples_file, "r") as f:
                self.mastered_samples = set(line.strip() for line in f if line.strip())
        
        sampler = FilteredRandomSampler(
            dataset, self.mastered_samples
        )

        super().__init__(dataset, sampler=sampler, *args, **kwargs)

    def add_mastered_sample(self, qid):
        if self.sampler and isinstance(self.sampler, FilteredRandomSampler):
            self.sampler.add_mastered_sample(qid)

        if self.mastered_samples_file:
            with open(self.mastered_samples_file, "a") as f:
                f.write(f"{qid}\n")

    def state_dict(self):
        state = super().state_dict()
        state["mastered_samples"] = list(self.mastered_samples)
        return state

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        if "mastered_samples" in state_dict:
            self.mastered_samples = set(state_dict["mastered_samples"])
            if self.sampler and isinstance(self.sampler, FilteredRandomSampler):
                self.sampler.mastered_samples = self.mastered_samples
                self.sampler._init_indices()
