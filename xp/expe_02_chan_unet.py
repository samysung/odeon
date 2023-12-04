import os

from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning import seed_everything
import albumentations as A
from typing import Dict

import os
import sys
sys.path.append('../odeon')

from odeon.data.data_module import Input
from odeon.models.change.module.change_unet import ChangeUnet

#root: str = '/media/HP-2007S005-data'
#root_dir: str = os.path.join(root, 'gers/change_dataset/patches')
root: str = '/mnt/stores/store-DAI/datasrc/dchan'

root_dir: str = os.path.join(root, 'gers/change/patches')
fold_nb: int = 0
fold: str = f'split-{fold_nb}'
root_fold: str = os.path.join(root_dir, fold)
dataset: str = os.path.join(root_fold, 'train_split_'+str(fold_nb)+'.geojson')
train_batch_size = 8
batch_size = 16
input_fields: Dict = {"T0": {"name": "T0", "type": "raster", "dtype": "uint8", "band_indices": [1, 2, 3]},
                               "T1": {"name": "T1", "type": "raster", "dtype": "uint8", "band_indices": [1, 2, 3]},
                               "mask": {"name": "change", "type": "mask", "encoding": "integer"}}

transform = [A.RandomRotate90(p=0.5),
            A.OneOf([A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5)], p=0.75)]
fit_params = {'input_fields': input_fields,
                               'dataloader_options' : {"batch_size": train_batch_size, "num_workers": 32, "shuffle": True},
                               'input_file': dataset,
                               'root_dir': root_dir,
                               'transform': transform
              } # add transform for data augment
val_dataset: str = os.path.join(root_fold, 'val_split_'+str(fold_nb)+'.geojson')
val_params = {'input_fields': input_fields,
                               'dataloader_options' : {"batch_size": batch_size, "num_workers": 32, "shuffle": False},
                               'input_file': val_dataset,
                               'root_dir': root_dir
              }
test_dataset: str = os.path.join(root_fold, 'test_split_'+str(fold_nb)+'.geojson')
test_params = {'input_fields': input_fields,
                               'dataloader_options' : {"batch_size": batch_size, "num_workers": 32, "shuffle": False},
                               'input_file': test_dataset,
                               'root_dir': root_dir
              }

input = Input(fit_params=fit_params,
              validate_params=val_params,
              test_params=test_params)
model = ChangeUnet(model='fc_siam_conc', scheduler='ExponentialLR', lr=0.001)
path_model_checkpoint = 'ckpt' # Need to specify by run, no ?
save_top_k_models = 5
path_model_log = ''
accelerator = 'gpu' # 'cpu'
limit_train_batches = 10
limit_val_batches = 10
limit_test_batches = 10
max_epochs = 100
check_val_every_n_epoch = 5
log_every_n_steps = 5
gpus = [0]
def main():
    seed_everything(42, workers=True)

    lr_monitor = LearningRateMonitor(logging_interval="step")
    model_checkpoint = ModelCheckpoint(dirpath=path_model_checkpoint,
                                       save_top_k=save_top_k_models,
                                       filename='epoch-{epoch}-loss-{val_bin_iou:.2f}',
                                       mode="max",
                                       monitor='val_bin_iou')
    callbacks = [lr_monitor, model_checkpoint]
    logger = pl_loggers.TensorBoardLogger(save_dir=path_model_log, version='test_exp2')
    trainer = Trainer(logger=logger, callbacks=callbacks, accelerator=accelerator, max_epochs=max_epochs,
                      log_every_n_steps=log_every_n_steps,gpus=gpus)
    trainer.fit(model=model, datamodule=input)
    trainer.validate(model=model, datamodule=input) # Where are stored the values ?
    trainer.test(model=model, datamodule=input)
    # Qualitative eval ?


if __name__ == '__main__':

    main()
