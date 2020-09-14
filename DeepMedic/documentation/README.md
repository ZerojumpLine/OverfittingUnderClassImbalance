DeepMedic
=====================================

## Requirements 

```
tensorflow-gpu==1.8.0
```

## Data and preprocssing
1. Download the data from [Brats](https://www.med.upenn.edu/sbia/brats2018/data.html) or [ATLAS](http://fcon_1000.projects.nitrc.org/indi/retro/atlas.html). 
2. (Optional but encouraged) Generate the brain masks, in most cases, masks generated based on intensity are good enough.
3. (Optional but encouraged) Normalize the intensity of the data within the ROI (brain masks) to a zero-mean, unit-variance space.
4. Making the datafile, referring to `./examples/configFiles/datafile`

## Train
Set the parameters in the training file (i.e. `./examples/configFiles/deepMedic/train/brats/trainConfig_brats_10percent_core_asymmetricfocal.cfg`) following the guidance there, and run the experiment.

For example, train brain tumor core segmentation with Brats using asymmetric focal loss using a basic DeepMedic setting, the command line is:

```
./deepMedicRun -model ./examples/configFiles/deepMedic/model/modelConfigbasic.cfg \
               -train ./examples/configFiles/deepMedic/train/brats/trainConfig_bratscore_10percent_basicsetting_asymmetricfocal.cfg -dev cuda0
```

For example, train brain lesion segmentation with ATLAS using asymmetric focal loss using a default DeepMedic setting, the command line is:

```
./deepMedicRun -model ./examples/configFiles/deepMedic/model/modelConfig1in.cfg \
               -train ./examples/configFiles/deepMedic/train/ATLAS/trainConfig_ATLAS_50percent_defaultsetting_asymmetricfocal.cfg -dev cuda0
```

## Modifications compared with original DeepMedic
Please check this files if you want integrate the functions in you own pipeline.

- Asymmetric large margin loss: mainly in   
    `./deepmedic/neuralnet/cost_functions.py`
- Asymmetric focal loss: mainly in  
    `./deepmedic/neuralnet/cost_functions.py`
- Asymmetric adversariall training: mainly in   
    `./deepmedic/neuralnet/cnn3d.py`    
    `./deepmedic/routines/training.py`
- Asymmetric mixup: mainly in   
    `./deepmedic/neuralnet/cost_functions.py`   
    `./deepmedic/routines/training.py`
- Asymmetric augmentation: mainly in    
    `./deepmedic/dataManagement/sampling.py`

## Citation
Please also consider to cite the original DeepoMedic paper when you find this useful:

```
@article{kamnitsas2017efficient,
  title={Efficient multi-scale 3D CNN with fully connected CRF for accurate brain lesion segmentation},
  author={Kamnitsas, Konstantinos and Ledig, Christian and Newcombe, Virginia FJ and Simpson, Joanna P and Kane, Andrew D and Menon, David K and Rueckert, Daniel and Glocker, Ben},
  journal={Medical image analysis},
  volume={36},
  pages={61--78},
  year={2017},
  publisher={Elsevier}
}
```
