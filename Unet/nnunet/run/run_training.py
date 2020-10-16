#    Copyright 2019 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

## env
import os
import torch
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
rootPath = os.path.split(rootPath)[0]
sys.path.append(rootPath)
import argparse
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.run.default_configuration import get_default_configuration
from nnunet.paths import default_plans_identifier
from nnunet.training.cascade_stuff.predict_next_stage import predict_next_stage
from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer
from nnunet.training.network_training.nnUNetTrainerCascadeFullRes import nnUNetTrainerCascadeFullRes

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("network")
    parser.add_argument("network_trainer")
    parser.add_argument("task")
    parser.add_argument("fold", help='0, 1, ..., 5 or \'all\'')
    parser.add_argument("-val", "--validation_only", help="use this if you want to only run the validation",
                        action="store_true")
    parser.add_argument("-c", "--continue_training", help="use this if you want to continue a training",
                        action="store_true")
    parser.add_argument("-p", help="plans identifier", default=default_plans_identifier, required=False)
    parser.add_argument("-u", "--unpack_data", help="Leave it as 1, development only", required=False, default=1,
                        type=int)
    parser.add_argument("--ndet", help="Per default training is deterministic, "
                                                   "nondeterministic allows cudnn.benchmark which will can give up to "
                                                   "20%% performance. Set this to do nondeterministic training",
                        required=False, default=False, action="store_true")
    parser.add_argument("--npz", required=False, default=False, action="store_true", help="if set then nnUNet will "
                                                                                          "export npz files of "
                                                                                          "predicted segmentations "
                                                                                          "in the vlaidation as well. "
                                                                                          "This is needed to run the "
                                                                                          "ensembling step so unless "
                                                                                          "you are developing nnUNet "
                                                                                          "you should enable this")
    parser.add_argument("--find_lr", required=False, default=False, action="store_true", help="not used here, just for fun")
    parser.add_argument("--valbest", required=False, default=False, action="store_true", help="hands off. This is not intended to be used")
    parser.add_argument("--fp16", required=False, default=False, action="store_true", help="enable fp16 training. Makes sense for 2d only! (and only on supported hardware!)")
    parser.add_argument('--asy', type=int, default=0, help='use asymmetric adversarial training, default 0 we use symmetric adversarial training, '
                                                           '1 for adversairal on the foreground, 2 for adversarial on the tumor class')
    parser.add_argument('--xi', type=float, default=1e-5, help='the hyperparameter for adversarial training, xi')
    parser.add_argument('--eps', type=float, default=10, help='the hyperparameter for adversarial training, eps')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
    parser.add_argument("-r", "--raw", help="if the adv / mixup is calculated with the raw image (w/0 augmentation)", action="store_true")
    parser.add_argument('--initlr', type=float, default=3e-4, help='gpu device id')
    parser.add_argument('--marginm', type=float, default=1, help='the hyperparameter for large margin loss, margin')
    parser.add_argument('--gama', type=float, default=4, help='the hyperparameter for focal loss, gama')
    parser.add_argument('--alpha', type=float, default=0.2, help='the hyperparameter for beta distribution, alpha')
    parser.add_argument('--margin', type=float, default=0.2, help='the hyperparameter for asymmetric mixup, margin')
    parser.add_argument('--weights', type=int, default=1, help='the weights for the tumor class')
    parser.add_argument("--wo_innerclass", help="mixup does not include in", action="store_true")
    parser.add_argument('--probBGGetAugment', type=float, default=1, help='the probabiltiy of background samples get augmented')

    args = parser.parse_args()
    torch.cuda.set_device(args.gpu)
    task = args.task
    fold = args.fold
    network = args.network
    network_trainer = args.network_trainer
    validation_only = args.validation_only
    plans_identifier = args.p
    find_lr = args.find_lr
    unpack = args.unpack_data
    deterministic = not args.ndet
    valbest = args.valbest
    fp16 = args.fp16
    asy = args.asy
    xi = args.xi
    eps = args.eps
    marginm = args.marginm
    gama = args.gama
    alpha = args.alpha
    margin = args.margin

    if unpack == 0:
        unpack = False
    elif unpack == 1:
        unpack = True
    else:
        raise ValueError("Unexpected value for -u/--unpack_data: %s. Use 1 or 0." % str(unpack))

    if args.wo_innerclass:
        inner = 0
    else:
        inner = 1

    if fold == 'all':
        pass
    else:
        fold = int(fold)

    plans_file, output_folder_name, dataset_directory, batch_dice, stage, \
        trainer_class = get_default_configuration(network, task, network_trainer, plans_identifier)


    output_folder_name = output_folder_name + 'asy' + str(args.asy) + 'xi' + str(args.xi) + 'eps' + str(args.eps) + 'RAWdice' + 'initlr' + str(args.initlr) \
                         + 'marginm' + str(args.marginm) + 'gama' + str(args.gama) + 'alpha' + str(args.alpha) + 'margin' + str(args.margin) + 'probBGGetAugment' + str(args.probBGGetAugment)

    if trainer_class is None:
        raise RuntimeError("Could not find trainer class in nnunet.training.network_training")

    if network == "3d_cascade_fullres":
        assert issubclass(trainer_class, nnUNetTrainerCascadeFullRes), "If running 3d_cascade_fullres then your " \
                                                                       "trainer class must be derived from " \
                                                                       "nnUNetTrainerCascadeFullRes"
    else:
        assert issubclass(trainer_class, nnUNetTrainer), "network_trainer was found but is not derived from nnUNetTrainer"

    trainer = trainer_class(plans_file, fold, output_folder=output_folder_name, dataset_directory=dataset_directory,
                            batch_dice=batch_dice, stage=stage, unpack_data=unpack, deterministic=deterministic,
                            fp16=fp16, asy = asy, xi = xi, eps = eps, raw=args.raw, initlr = args.initlr, marginm = marginm,
                            gama = gama, alpha = alpha, margin = margin, weights = args.weights, inner = inner, probBGGetAugment = args.probBGGetAugment)

    trainer.initialize(not validation_only)

    if find_lr:
        trainer.find_lr()
    else:
        if not validation_only:
            if args.continue_training:
                trainer.load_latest_checkpoint()
            trainer.run_training()
        elif not valbest:
            trainer.load_latest_checkpoint(train=False)

        if valbest:
            trainer.load_best_checkpoint(train=False)
            val_folder = "validation_best_epoch"
        else:
            val_folder = "validation"

        # predict validation
        trainer.validate(save_softmax=args.npz, validation_folder_name=val_folder)

        if network == '3d_lowres':
            trainer.load_best_checkpoint(False)
            print("predicting segmentations for the next stage of the cascade")
            predict_next_stage(trainer, join(dataset_directory, trainer.plans['data_identifier'] + "_stage%d" % 1))
