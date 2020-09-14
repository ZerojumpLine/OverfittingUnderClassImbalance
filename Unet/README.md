nnU-Net
=====================================

## Requirements 

```
pytorch==1.1.0
```

## Data and preprocssing
1. Download the data from [KiTS19](https://github.com/neheller/kits19).
2. (Optional) Resample the datasest to a uniform space with 1.6mm * 1.6mm * 3.2mm
3. Convert the dataset following nnU-Net's [instructions](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_conversion.md). I name the KiTS dataset as Task01_Kits.
4. Run the preprocessing with 
    ```
    python experiment_planning/plan_and_preprocess_task.py -t Task01_Kits
    ```
5. Set your path in `./nnunet/paths.py`. Basically, set *base* as where you set your dataset.


## Train
- (Optional) let nnU-Net create the default split file (by running one epoch but stop), change split in  'preprocessing_output_dir/Task01_Kits/splits_final.pkl'. I set different folders with different portion of training data. Folder 0 stands for 100%, folder 2 stands for 50% and folder 3 stands for 10%.


Run nnUnet with 10% Kits data with asymmetric focal loss:

```
python run/run_training.py 3d_fullres nnUNetTrainer Task01_Kits 3 --ndet --asy 2 --marginm 0 --gama 4 --xi 1e-5 --eps 0 --margin 0 --alpha 1 --probBGGetAugment 1 --raw --gpu 0
```

- Parameter "asy": Indicates the version of regularization techniques. 0 stands for symmetric regulalrizations, 1 stands for asymmetric variants aiming at improving both foreground cls, 2 stands for asymmetric variants, aiming at only improving cls 2.
- Parameter "marginm": The hyperpamrameter for the large margin loss. 0.2 ~ 2.0 is reasonable.
- Parameter "gama": The hyperparameter for focal loss. 2.0 ~ 6.0 is the reasonbale range.
- Parameter "xi": The hyperparamter for initial perturbs. 10**(-5) is good. It is not that sensitive
- Parameter "eps": The hyperparamter for adversarial training indicating the magnitude of adversrial perturbs. 10 is a good choice. Set it to 0 to disable adversarial training.
- Parameter "margin": The hyperpamrameter for asymmetric mixup, set the margin to remain the background class. Set it to 0 to disable mixup.
- Parameter "alpha": The hyperpamrameter for the mixup, draw the weights from the beta distribution Beta(mixuprate, mixuprate).
- Parameter "raw": If the adversairal training / mixup are calculated with the raw image (w/o augmentation). set it as true can be more stable considering that the default augmentation is very strong.
- Parameter "probBGGetAugment": How much probabiltiy the background cls gets augmented.

## Modifications compared with original nnU-Net
Please check this files if you want integrate the functions in you own pipeline.

- Asymmetric large margin loss: mainly in   
    `./nnunet/training/loss_functions/mND_Crossentropy.py`
- Asymmetric focal loss: mainly in  
    `./nnunet/training/loss_functions/mND_Crossentropy.py`  
    `./nnunet/training/loss_functions/dice_loss.py`
- Asymmetric adversariall training: mainly in   
    `./nnunet/training/network_training/network_trainer.py/run_iterationRAW`    
    `./nnunet/training/loss_functions/advND_Crossentropy.py`
- Asymmetric mixup: mainly in   
    `./nnunet/training/network_training/network_trainer.py/run_iterationRAW`    
    `./nnunet/training/loss_functions/mixND_Crossentropy.py`
- Asymmetric augmentation: mainly in    
    `./nnunet/training/data_augmentation/default_data_augmentation.py/get_default_augmentation`     
    `./nnunet/training/network_training/network_trainer.py/run_iterationRAW`

## Citation
Please also consider to cite the original nnU-Net paper when you find this useful:

```
@article{isensee2019automated,
  title={Automated design of deep learning methods for biomedical image segmentation},
  author={Isensee, Fabian and J{\"a}ger, Paul F and Kohl, Simon AA and Petersen, Jens and Maier-Hein, Klaus H},
  journal={arXiv preprint arXiv:1904.08128},
  year={2019}
}
```