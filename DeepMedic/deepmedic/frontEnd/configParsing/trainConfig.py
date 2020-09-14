# Copyright (c) 2016, Konstantinos Kamnitsas
# All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the BSD license. See the accompanying LICENSE file
# or read the terms at https://opensource.org/licenses/BSD-3-Clause.

from __future__ import absolute_import, print_function, division
import os

from deepmedic.frontEnd.configParsing.utils import *
from deepmedic.frontEnd.configParsing.config import Config


class TrainConfig(Config):
    
    #Optional but highly suggested.
    SESSION_NAME = "sessionName"
    #[REQUIRED]
    FOLDER_OUTP = "folderForOutput" #MUST BE GIVEN
    SAVED_MODEL = "cnnModelFilePath" #MUST BE GIVEN
    
    #=============TRAINING========================
    CHANNELS_TR = "channelsTraining" #MUST BE GIVEN
    GT_LABELS_TR = "gtLabelsTraining"
    ROI_MASKS_TR = "roiMasksTraining"
    
    # DEPRECATED! parsed for backwards compatibility and providing a warning! Use advanced sampling options instead.
    PERC_POS_SAMPLES_TR = "percentOfSamplesToExtractPositiveTrain"
    
    #~~~~Advanced Sampling~~~~~
    DEFAULT_TR_SAMPLING = "useDefaultTrainingSamplingFromGtAndRoi"
    TYPE_OF_SAMPLING_TR = "typeOfSamplingForTraining"
    PROP_OF_SAMPLES_PER_CAT_TR = "proportionOfSamplesToExtractPerCategoryTraining"
    WEIGHT_MAPS_PER_CAT_FILEPATHS_TR = "weightedMapsForSamplingEachCategoryTrain"
    
    #~~~~~ Training cycle ~~~~~~~~
    NUM_EPOCHS = "numberOfEpochs"
    NUM_SUBEP = "numberOfSubepochs"
    NUM_CASES_LOADED_PERSUB = "numOfCasesLoadedPerSubepoch"
    NUM_TR_SEGMS_LOADED_PERSUB = "numberTrainingSegmentsLoadedOnGpuPerSubep"
    NUM_OF_PROC_SAMPL = "num_processes_sampling"
    #~~~~~ Learning rate schedule ~~~~~
    LR_SCH_TYPE = "typeOfLearningRateSchedule"
    #Stable + Auto + Predefined.
    DIV_LR_BY = "whenDecreasingDivideLrBy"
    #Stable + Auto
    NUM_EPOCHS_WAIT = "numEpochsToWaitBeforeLoweringLr"
    #Auto:
    AUTO_MIN_INCR_VAL_ACC = "min_incr_of_val_acc_considered"
    #Predefined.
    PREDEF_SCH = "predefinedSchedule"
    #Exponential
    EXPON_SCH = "paramsForExpSchedForLrAndMom"
    #~~~~ Data Augmentation~~~
    REFL_AUGM_PER_AXIS = "reflectImagesPerAxis"
    PERF_INT_AUGM_BOOL = "performIntAugm"
    INT_AUGM_SHIF_MUSTD = "sampleIntAugmShiftWithMuAndStd"
    INT_AUGM_MULT_MUSTD = "sampleIntAugmMultiWithMuAndStd"
    
    #============== VALIDATION ===================
    PERFORM_VAL_SAMPLES = "performValidationOnSamplesThroughoutTraining"
    PERFORM_VAL_INFERENCE = "performFullInferenceOnValidationImagesEveryFewEpochs"
    CHANNELS_VAL = "channelsValidation"
    GT_LABELS_VAL = "gtLabelsValidation"
    #For SAMPLES Validation:
    NUM_VAL_SEGMS_LOADED_PERSUB = "numberValidationSegmentsLoadedOnGpuPerSubep"
    #Optional
    ROI_MASKS_VAL = "roiMasksValidation"
    #~~~~~~~~~Full Inference~~~~~~~~
    NUM_EPOCHS_BETWEEN_VAL_INF = "numberOfEpochsBetweenFullInferenceOnValImages"
    NAMES_FOR_PRED_PER_CASE_VAL = "namesForPredictionsPerCaseVal"
    SAVE_SEGM_VAL = "saveSegmentationVal"
    SAVE_PROBMAPS_PER_CLASS_VAL = "saveProbMapsForEachClassVal"
    SUFFIX_SEGM_PROB_VAL = "suffixForSegmAndProbsDictVal"
    SAVE_INDIV_FMS_VAL = "saveIndividualFmsVal"
    SAVE_4DIM_FMS_VAL = "saveAllFmsIn4DimImageVal"
    INDICES_OF_FMS_TO_SAVE_NORMAL_VAL = "minMaxIndicesOfFmsToSaveFromEachLayerOfNormalPathwayVal"
    INDICES_OF_FMS_TO_SAVE_SUBSAMPLED_VAL = "minMaxIndicesOfFmsToSaveFromEachLayerOfSubsampledPathwayVal"
    INDICES_OF_FMS_TO_SAVE_FC_VAL = "minMaxIndicesOfFmsToSaveFromEachLayerOfFullyConnectedPathwayVal"
    #~~~~~~~~Advanced Validation Sampling~~~~~~~~~~~~
    DEFAULT_VAL_SAMPLING = "useDefaultUniformValidationSampling"
    TYPE_OF_SAMPLING_VAL = "typeOfSamplingForVal"
    PROP_OF_SAMPLES_PER_CAT_VAL = "proportionOfSamplesToExtractPerCategoryVal"
    WEIGHT_MAPS_PER_CAT_FILEPATHS_VAL = "weightedMapsForSamplingEachCategoryVal"
    
    #====OPTIMIZATION=====
    LRATE = "learningRate"
    OPTIMIZER = "sgd0orAdam1orRms2"
    MOM_TYPE = "classicMom0OrNesterov1"
    MOM = "momentumValue"
    MOM_NORM_NONNORM = "momNonNorm0orNormalized1"
    N_EPS = "Nepsilon"
    N_XI = "Nxi"
    N_SIDE = "Nside"
    #Adam
    B1_ADAM = "b1Adam"
    B2_ADAM = "b2Adam"
    EPS_ADAM = "epsilonAdam"
    #RMS
    RHO_RMS = "rhoRms"
    EPS_RMS = "epsilonRms"
    #Losses
    LOSSES_WEIGHTS = "losses_and_weights"
    W_C_IN_COST = "reweight_classes_in_cost"
    #Regularization L1 and L2.
    L1_REG = "L1_reg"
    L2_REG = "L2_reg"

    MARGIN = "marginm"
    MIX_RATE = "mixuprate"
    MIX_MAR = "mixupbiasmargin"

    PROB_AUGBG = "probaugmentbackground"
    
    #~~~  Freeze Layers ~~~
    LAYERS_TO_FREEZE_NORM = "layersToFreezeNormal"
    LAYERS_TO_FREEZE_SUBS = "layersToFreezeSubsampled"
    LAYERS_TO_FREEZE_FC = "layersToFreezeFC"
    
    #========= GENERICS =========
    PAD_INPUT = "padInputImagesBool"
    RUN_INP_CHECKS = "run_input_checks"
    

    def __init__(self, abs_path_to_cfg):
        Config.__init__(self, abs_path_to_cfg)
        
    
    #If certain config args are given in command line, completely override the corresponding ones in the config files.
    def override_file_cfg_with_cmd_line_cfg(self, log, args):
        if args.saved_model is not None:
            abs_path_model_cmd_line = getAbsPathEvenIfRelativeIsGiven(args.saved_model, os.getcwd())
            if self.get( self.SAVED_MODEL ) is not None:
                log.print3("WARN: A model to load was specified both in the command line and in the train-config file!\n"+\
                            "\t The input by the command line will be used: " + str(abs_path_model_cmd_line) )
            
            self._configStruct[ self.SAVED_MODEL ] = abs_path_model_cmd_line
            
            










