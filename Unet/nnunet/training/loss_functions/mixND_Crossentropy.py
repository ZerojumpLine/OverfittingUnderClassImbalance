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

import torch
from torch import nn
import torch.nn.functional as F

class mixCrossentropyND(nn.Module):
    """
    Network has to have NO NONLINEARITY!
    """
    def __init__(self, asy, margin, inner, gama, marginm, weights):
        super(mixCrossentropyND, self).__init__()
        self.asy = asy
        self.margin = margin
        self.inner = inner
        self.gama = gama
        self.marginm = marginm
        self.weights = weights

    def forward(self, inp, target, lam):
        targetmix = target.flip(0)
        target = target.long()
        targetmix = targetmix.long()

        num_classes = inp.size()[1]

        i0 = 1
        i1 = 2

        while i1 < len(inp.shape): # this is ugly but torch only allows to transpose two axes at once
            inp = inp.transpose(i0, i1)
            i0 += 1
            i1 += 1

        inp = inp.contiguous()
        inp = inp.view(-1, num_classes)

        target = target.view(-1,)
        targetmix = targetmix.view(-1,)

        # now inp is [N,C], target is [N,]
        ################################### Symmetric/ asymmetric mixup ###################################

        if self.asy == 0:
            y_one_hot = self.one_hot_embedding(target.data.cpu(), num_classes)
            y_one_hotmix = self.one_hot_embedding(targetmix.data.cpu(), num_classes)
            y_one_hotmixup = lam * y_one_hot + (1 - lam) * y_one_hotmix
            ydsclossposition = y_one_hotmixup[:, 0] < 100  # everything

        if self.asy == 1:  # mixup for the both foreground
            backgroundlabel = 0
            # once it is backgr component, take it as another thing
            y_comb0 = torch.where(target == backgroundlabel, targetmix, target)
            y_comb1 = torch.where(targetmix == backgroundlabel, target, targetmix)
            # if the background mixup lambda is less than the margin, set as another component
            if lam < self.margin:
                y_one_hot = self.one_hot_embedding(y_comb0.data.cpu(), num_classes)
            else:
                y_one_hot = self.one_hot_embedding(target.data.cpu(), num_classes)
            if 1 - lam < self.margin:
                y_one_hotmix = self.one_hot_embedding(y_comb1.data.cpu(), num_classes)
            else:
                y_one_hotmix = self.one_hot_embedding(targetmix.data.cpu(), num_classes)
            y_one_hotmixup = lam * y_one_hot + (1 - lam) * y_one_hotmix
            # discard the mixed samples, which is a combination of kidney and tumor, or a large backgr with a kidney/ tumor
            y_one_hotmixup = torch.floor(y_one_hotmixup)

            ydsclosspositionc0 = (y_one_hotmixup[:, 0] > 0) & (y_one_hotmixup[:, 0] < 1)
            ydsclosspositionc1 = (y_one_hotmixup[:, 1] > 0) & (y_one_hotmixup[:, 1] < 1)
            ydsclosspositionc2 = (y_one_hotmixup[:, 2] > 0) & (y_one_hotmixup[:, 2] < 1)
            ydsclossposition = ydsclosspositionc0 + ydsclosspositionc1 + ydsclosspositionc2
            ydsclossposition = 1 - ydsclossposition

        if self.asy == 2:# mixup for the both foreground
            # once there are tumour component, take it as tumour
            tumorlabel = 2
            y_comb0 = torch.where(targetmix == tumorlabel, targetmix, target)
            y_comb1 = torch.where(target == tumorlabel, target, targetmix)

            # here it is a littler dumpy, considering that CT background intensity is too low
            # I dont mix tumor with background.
            # y_comb0 = torch.where(target != 0, y_comb0, target)
            # y_comb1 = torch.where(targetmix != 0, y_comb1, targetmix)

            # if the another component mixup lambda is less than the margin, set as tumor
            if lam < self.margin:
                y_one_hot = self.one_hot_embedding(y_comb0.data.cpu(), num_classes)
            else:
                y_one_hot = self.one_hot_embedding(target.data.cpu(), num_classes)
            if 1-lam < self.margin:
                y_one_hotmix = self.one_hot_embedding(y_comb1.data.cpu(), num_classes)
            else:
                y_one_hotmix = self.one_hot_embedding(targetmix.data.cpu(), num_classes)
            y_one_hotmixup = lam * y_one_hot + (1 - lam) * y_one_hotmix
            # if the sample is generated by mixup, I do not want middle results, such as "0.3 tumor"
            # have no loss when background and kidney are mixed, or tumor is small
            # but it could cause trouble for sample dice loss? in theory it would not
            # it might have problems, gradients are backwards for (theta/ sum(p) + theta)
            # I should save the loss position
            ydsclosspositionc0 = (y_one_hotmixup[:, 0] > 0) & (y_one_hotmixup[:, 0] < 1)
            ydsclosspositionc1 = (y_one_hotmixup[:, 1] > 0) & (y_one_hotmixup[:, 1] < 1)
            ydsclosspositionc2 = (y_one_hotmixup[:, 2] > 0) & (y_one_hotmixup[:, 2] < 1)
            ydsclossposition = ydsclosspositionc0 + ydsclosspositionc1 + ydsclosspositionc2
            ydsclossposition = 1 - ydsclossposition
            y_one_hotmixup = torch.floor(y_one_hotmixup)

        y_one_hotmixup = y_one_hotmixup.cuda()

        #########################################################################################################
        # p_y_given_x_train is corresponding [N,C]

        ################################### Symmetric/ asymmetric large margin loss ###################################

        if self.inner == 0:
            # in this case, we exclude the inner class mixup
            # place_inner is [N,], 0 indicating the place where target and targetmix have the same label
            place_inner = target != targetmix
            place_inner = place_inner.float()
            place_inner.unsqueeze_(-1)
            # print('I am here')
            # print(place_inner.shape)
            # print(y_one_hotmixup.shape)
            y_one_hotmixup = y_one_hotmixup * place_inner

        # put the margin/focal from here.
        y_one_hotmixup[:, 2] = y_one_hotmixup[:, 2] * self.weights
        # p_y_given_x_train is corresponding [N,C]

        if self.asy == 0:
            # have margin on all classes.
            inpmargin = torch.stack((inp[:, 0] - self.margin, inp[:, 1] - self.margin, inp[:, 2] - self.margin), 1)
            # preserve the other sums
            inppost = torch.stack((inp[:, 0] * (target != 0).float() + inpmargin[:, 0] * (target == 0).float(),
                                   inp[:, 1] * (target != 1).float() + inpmargin[:, 1] * (target == 1).float(),
                                   inp[:, 2] * (target != 2).float() + inpmargin[:, 2] * (target == 2).float()), 1)
        if self.asy == 1:
            # have margin only on foreground class
            inpmargin = torch.stack((inp[:, 0], inp[:, 1] - self.margin, inp[:, 2] - self.margin), 1)
            # preserve the other sums
            inppost = torch.stack((inp[:, 0] * (target != 0).float() + inpmargin[:, 0] * (target == 0).float(),
                                   inp[:, 1] * (target != 1).float() + inpmargin[:, 1] * (target == 1).float(),
                                   inp[:, 2] * (target != 2).float() + inpmargin[:, 2] * (target == 2).float()), 1)
        if self.asy == 2:
            # have margin only on the tumor class
            inpmargin = torch.stack((inp[:, 0], inp[:, 1], inp[:, 2] - self.marginm), 1)
            # preserve the other sums
            inppost = torch.stack((inp[:, 0] * (target != 0).float() + inpmargin[:, 0] * (target == 0).float(),
                                   inp[:, 1] * (target != 1).float() + inpmargin[:, 1] * (target == 1).float(),
                                   inp[:, 2] * (target != 2).float() + inpmargin[:, 2] * (target == 2).float()), 1)

        p_y_given_x_train = torch.softmax(inppost, 1)
        e1 = 1e-6  ## without the small margin, it would lead to nan after several epochs
        log_p_y_given_x_train = (p_y_given_x_train + e1).log()
        #########################################################################################################

        ################################### Symmetric/ asymmetric focal loss ###################################

        focal_conduct_active1 = (1 - p_y_given_x_train + e1) ** self.gama
        focal_conduct_active2 = (p_y_given_x_train + e1) ** self.gama

        if self.asy == 0:
            # have focal reduction on all classes.
            focal_conduct1 = focal_conduct_active1
            focal_conduct2 = focal_conduct_active2
        if self.asy == 1:
            # have focal reduction only on 0 class
            conduct_ones = torch.ones(p_y_given_x_train.size()[0], 2)
            conduct_ones = conduct_ones.cuda()
            focal_conduct1 = torch.cat((focal_conduct_active1[:, 0:1], conduct_ones), 1)
            focal_conduct2 = torch.cat((focal_conduct_active2[:, 0:1], conduct_ones), 1)
        if self.asy == 2:
            # have focal reduction only on 0 and 1 class
            conduct_ones = torch.ones(p_y_given_x_train.size()[0], 1)
            conduct_ones = conduct_ones.cuda()
            focal_conduct1 = torch.cat((focal_conduct_active1[:, 0:2], conduct_ones), 1)
            focal_conduct2 = torch.cat((focal_conduct_active2[:, 0:2], conduct_ones), 1)

        m_log_p_y_given_x_train = focal_conduct1 * log_p_y_given_x_train

        num_samples = m_log_p_y_given_x_train.size()[0]

        loss = - (1. / num_samples) * m_log_p_y_given_x_train * y_one_hotmixup

        #########################################################################################################

        # print(loss.sum())
        # print(F.cross_entropy(inp, target)) # correct and original one
        return loss.sum(), y_one_hotmixup, ydsclossposition, inppost, focal_conduct1, focal_conduct2


    def one_hot_embedding(self, labels, num_classes):
        '''Embedding labels to one-hot form.
        Args:
          labels: (LongTensor) class labels, sized [N,].
          num_classes: (int) number of classes.
        Returns:
          (tensor) encoded labels, sized [N,#classes].
        '''
        y = torch.eye(num_classes)  # [D,D]

        return y[labels]            # [N,D]
