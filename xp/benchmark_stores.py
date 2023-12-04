from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, Tuple
from logging import getLogger
from resource import getrusage as resource_usage, RUSAGE_SELF
from time import time as timestamp
import json


from tqdm import tqdm
from multiprocessing import cpu_count
from jsonargparse import ArgumentParser, CLI, Namespace, SignatureArguments

# from odeon import Input
from odeon.core.types import STAGES_OR_VALUE
from odeon.data.core.dataloader_utils import DEFAULT_DATALOADER_OPTIONS
from odeon.data.stage import DataFactory

logger = getLogger(__name__)
# local: /var/data/dl
# store dai: /mnt/stores/store-DAI/datasrc/dchan
# store flechette: /mnt/stores/SMLPRIINFSAP1


@dataclass
class BenchmarkStores:
    store: str = 'store-DAI'
    input_fields: Dict | None = None
    store_dir: str = '/mnt/stores/store-DAI/datasrc/dchan'
    root_dir: str | None = None
    input_file: str | None = None
    output_dir: str = '/mnt/stores/store-DAI/equipiers/skhelifi/benchmark_stores'
    stage: STAGES_OR_VALUE = 'fit'
    n_cycle: int = 20
    transform: None | Dict[str, Any] = None
    dataloader_options: None | Dict[str, Any] = None
    input_files_has_header = True
    by_zone: bool = False
    patch_size: Tuple | None = None  # (256, 256)
    patch_resolution: Tuple | None = None  # (0.2, 0.2)
    random_window: bool = True
    overlap: Tuple | None = None  # (0.0, 0.0)
    cache_dataset: bool = True
    debug: bool = False
    cpu_count: None | int = None

    def __post_init__(self):
        self.root_dir = str(Path(self.store_dir) / "gers/change_dataset/patches")
        self.input_file = str(Path(self.root_dir) / "split-0/train_split_0.geojson")
        self.input_fields = {"T0": {"name": "T0", "type": "raster", "dtype": "uint8", "band_indices": [1, 2, 3, 4, 5]},
                             "T1": {"name": "T1", "type": "raster", "dtype": "uint8", "band_indices": [1, 2, 3, 4, 5]},
                             "mask": {"name": "change", "type": "mask", "encoding": "integer"}}
        if self.cpu_count is None:
            self.cpu_count = int(cpu_count())

        if self.dataloader_options is None:
            self.dataloader_options = {"batch_size": 16, "num_workers": self.cpu_count, "shuffle": True}

    def __call__(self, *args, **kwargs):
        self.run(*args, **kwargs)

    def run(self, *args, **kwargs):
        start_time, start_resources = timestamp(), resource_usage(RUSAGE_SELF)

        data_loader, dataset, transform, dataframe = DataFactory.build_data(input_fields=self.input_fields,
                                                                            input_file=self.input_file,
                                                                            stage=self.stage,
                                                                            transform=self.transform,
                                                                            dataloader_options=self.dataloader_options,
                                                                            root_dir=self.root_dir,
                                                                            input_files_has_header=self.input_files_has_header,
                                                                            by_zone=self.by_zone,
                                                                            patch_size=self.patch_size,
                                                                            patch_resolution=self.patch_resolution,
                                                                            random_window=self.random_window,
                                                                            overlap=self.overlap,
                                                                            cache_dataset=self.cache_dataset,
                                                                            debug=self.debug)
        n_cycle = self.n_cycle
        logs = dict()
        logs['store'] = str(self.store)
        logs['time'] = dict()
        logger.info(f'number of cycle: {n_cycle}')
        logger.info(f'number of cpu process: {self.cpu_count}')
        total_time: float = 0.0
        for i in range(n_cycle):
            start = timestamp()
            for idx, batch in tqdm(enumerate(data_loader),
                                   total=int(len(dataframe)//self.dataloader_options['batch_size'])):
                pass
            end = timestamp()
            epoch_time = end - start
            logger.info(epoch_time)

            logs['time'][f'epoch-{i}'] = float(epoch_time)
            total_time = float(total_time + epoch_time)
        logs['total_time'] = total_time
        end_resources, end_time = resource_usage(RUSAGE_SELF), timestamp()

        logs['real'] = end_time - start_time,
        logs['sys'] = end_resources.ru_stime - start_resources.ru_stime
        logs['user'] = end_resources.ru_utime - start_resources.ru_utime

        with open(str(Path(self.output_dir) / f'test_{self.store}_{str(self.n_cycle)}_{str(self.cache_dataset)}.json'),
                  'w') as fp:
            json.dump(logs, fp)

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_dataclass_arguments(theclass=BenchmarkStores, nested_key='conf')
    cfg = parser.parse_args()
    cfg = parser.instantiate_classes(cfg=cfg)
    benchmark_stores = cfg.conf
    benchmark_stores()
