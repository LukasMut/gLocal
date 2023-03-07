from typing import Iterator

from torch.utils.data import DataLoader


class ZippedBatchLoader(DataLoader):
    def __init__(
        self,
        batches_i: Iterator,
        batches_j: Iterator,
        repeat_times: int = None,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last: bool = False,
    ) -> None:
        super().__init__(
            dataset=None,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )
        self.batches_i = batches_i
        self.batches_j = batches_j
        self.times = repeat_times

    @staticmethod
    def _repeat(object: Iterator, times: int = None) -> Iterator:
        """Either repeat the iterator <times> times or repeat it infinitely many times."""
        if times is None:
            while True:
                for x in object:
                    yield x
        else:
            for _ in range(times):
                for x in object:
                    yield x

    def _zip_batches(self) -> zip:
        """Zip ImageNet and THINGS batches into a single zipped batch iterator."""
        if len(self.batches_j) > len(self.batches_i):
            batches_i_repeated = self._repeat(self.batches_i, self.times)
            zipped_batches = zip(batches_i_repeated, self.batches_j)
        elif len(self.batches_j) < len(self.batches_i):
            batches_j_repeated = self._repeat(self.batches_j, self.times)
            zipped_batches = zip(self.batches_i, batches_j_repeated)
        else:
            zipped_batches = zip(self.batches_i, self.batches_j)
        return zipped_batches

    def __iter__(self) -> Iterator:
        return self._zip_batches()

    def __len__(self) -> int:
        if len(self.batches_j) > len(self.batches_i):
            length = len(self.batches_j)
        elif len(self.batches_j) < len(self.batches_i):
            length = len(self.batches_i)
        else:
            length = len(self.batches_j)
        return length
