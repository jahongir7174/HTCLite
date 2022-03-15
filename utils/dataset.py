from mmcv.utils import build_from_cfg
from mmdet.datasets import CocoDataset
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.pipelines import Compose

from utils.util import *


@DATASETS.register_module()
class INSDataset(CocoDataset):
    CLASSES = ('human', 'ball')


@DATASETS.register_module()
class MOSAICDataset:
    def __init__(self, dataset, image_size, pipeline):
        self.dataset = dataset
        self.CLASSES = dataset.CLASSES
        self.pipeline = Compose(pipeline)
        if hasattr(self.dataset, 'flag'):
            self.flag = numpy.zeros(len(dataset), dtype=numpy.uint8)
        self.image_sizes = image_size
        self.num_samples = len(dataset)
        self.indices = range(len(dataset))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        while True:
            if numpy.random.rand() > 0.5:
                data = mosaic(self, index)
            else:
                data = mix_up(self, index, random.choice(self.indices))
            if data is None:
                index = random.choice(self.indices)
                continue
            return process(self, data)


def build_dataset(cfg, default_args=None):
    if cfg['type'] == 'MOSAICDataset':
        cp_cfg = deepcopy(cfg)
        cp_cfg['dataset'] = build_dataset(cp_cfg['dataset'])
        cp_cfg.pop('type')
        return MOSAICDataset(**cp_cfg)
    else:
        return build_from_cfg(cfg, DATASETS, default_args)
