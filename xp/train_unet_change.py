import os

from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning import seed_everything
import albumentations as A
from typing import Dict

from datetime import datetime

import os
import sys
sys.path.append('../odeon')

from odeon.data.data_module import Input
from odeon.models.change.module.change_unet import ChangeUnet

#root: str = '/media/HP-2007S005-data'
#root_dir: str = os.path.join(root, 'gers/change_dataset/patches')
root: str = '/home/NGonthier/Documents/Detection_changement/data/'
if not os.path.exists(root):
    root: str = '/home/dl/gonthier/data/'
root_dir: str = os.path.join(root, 'gers/change/patches')
fold_nb: int = 0
fold: str = f'split-{fold_nb}'
root_fold: str = os.path.join(root_dir, fold)
dataset: str = os.path.join(root_fold, 'train_split_'+str(fold_nb)+'.geojson')
batch_size = 8
transform = [A.RandomRotate90(p=0.5),
            A.OneOf([A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5)], p=0.75)]
transform_name = 'Rot_Flip'
input_fields : Dict = {"T0": {"name": "T0", "type": "raster", "dtype": "uint8", "band_indices": [1, 2, 3]},
                               "T1": {"name": "T1", "type": "raster", "dtype": "uint8", "band_indices": [1, 2, 3]},
                               "mask": {"name": "change", "type": "mask", "encoding": "integer"}}
fit_params = {'input_fields': input_fields,
                               'dataloader_options' : {"batch_size": batch_size, "num_workers": 8},
                               'input_file': dataset,
                               'root_dir': root_dir,
                               'transform': transform
              }
val_dataset: str = os.path.join(root_fold, 'val_split_'+str(fold_nb)+'.geojson')
val_params = {'input_fields': input_fields,
                               'dataloader_options' : {"batch_size": batch_size, "num_workers": 8},
                               'input_file': val_dataset,
                               'root_dir': root_dir
              }
test_dataset: str = os.path.join(root_fold, 'test_split_'+str(fold_nb)+'.geojson')
test_params = {'input_fields': input_fields,
                               'dataloader_options' : {"batch_size": batch_size, "num_workers": 8},
                               'input_file': test_dataset,
                               'root_dir': root_dir
              }

input = Input(fit_params=fit_params,
              validate_params=val_params,
              test_params=test_params)
model_name = 'fc_siam_conc'
scheduler = 'ExponentialLR'
lr = 0.001
model_params: Dict = {'decoder_use_batchnorm': True, 'activation': "sigmoid", 'encoder_weights': None}
model = ChangeUnet(model=model_name, scheduler=scheduler, lr=lr, model_params=model_params)
path_model_checkpoint = 'ckpt' # Need to specify by run, no ?
save_top_k_models = 3
path_model_log = ''
accelerator = 'gpu' # 'cpu'
max_epochs = 500
check_val_every_n_epoch = 10
dt = datetime.now()# Getting the current date and time
time_tag = str(dt.month) + '-' + str(dt.day) + '-' + str(dt.hour) + '-' + str(dt.minute)
model_tag = time_tag + '_' + model_name + '_' + scheduler + '_lr'+str(lr) + '_'+transform_name+'_None'
def main():
    seed_everything(42, workers=True)

    lr_monitor = LearningRateMonitor(logging_interval="step") # Mettre un learning rate sinusoidale
    # ou bien comme dans le papier de Rodrigo : scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95)
    model_checkpoint = ModelCheckpoint(dirpath=path_model_checkpoint,
                                       save_top_k=save_top_k_models,
                                       filename=model_tag+'_epoch-{epoch}-loss-{val_bin_iou:.2f}',
                                       mode="max",
                                       monitor='val_bin_iou')
    early_stop = EarlyStopping(monitor="val_bin_iou", mode="max", patience=50, check_finite=True)
    #Faire un callback pour sauver les images
    callbacks = [lr_monitor, model_checkpoint, early_stop]
    logger = pl_loggers.TensorBoardLogger(save_dir=path_model_log, version=model_tag)
    trainer = Trainer(logger=logger, callbacks=callbacks, accelerator=accelerator, max_epochs=max_epochs)
    trainer.fit(model=model, datamodule=input)
    trainer.validate(model=model, datamodule=input) # Where are stored the values ?
    trainer.test(model=model, datamodule=input)
    # Qualitative eval ?


if __name__ == '__main__':

    main()
