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
        '''
        Be careful of the two margin (one for large margin loss, one for mixup), do not mess it up.
        specifically, margin is for mixup, marginm is for large margin loss.
        '''
        self.asy = asy
        self.margin = 1 - margin
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
        # inp is network_output

        target = target.view(-1,)
        targetmix = targetmix.view(-1,)

        # now inp is [N,C], target is [N,]
        ################################### Symmetric/ asymmetric mixup ###################################
        '''
        For mixup, I need the output y_one_hotmixup, with shape [N, C]
        I get two sample as y_one_hot/ y_one_hotmix
        and ydsclossposition indicating where y_one_hot == 1, there are some cases with probability < 1
        it is because the rare class samples do not have large enough components, I want to get rid of this cases. 
        '''

        if self.asy == 0:
            # just normal mixup, nothing special
            r = [1, 1, 1]
            y_one_hot = self.one_hot_embedding(target.data.cpu(), num_classes)
            y_one_hotmix = self.one_hot_embedding(targetmix.data.cpu(), num_classes)
            y_one_hotmixup = lam * y_one_hot + (1 - lam) * y_one_hotmix
            ydsclossposition = y_one_hotmixup[:, 0] < 100  # everything
        else:
            if self.asy == 1:  # mixup for the both foreground
                r = [0, 1, 1]

            if self.asy == 2:# mixup for the both foreground
                # once there are tumour component, take it as tumour
                r = [0, 0, 1]

            # I should consider the y_comb
            # this is the case, when this is any possible the the label of y should change
            rall = [i for i, e in enumerate(r) if e == 1]
            y_comb0 = target
            y_comb1 = targetmix
            # if the other one is taken as one of the rare classes, the combination should change
            for rindex in rall:
                y_comb0 = torch.where(targetmix == rindex, targetmix, y_comb0)
                y_comb1 = torch.where(target == rindex, target, y_comb1)

            # if there are several rare classes, we dont want to mix them
            # keep the y_comb as original labels, when they are rare classes.
            for rindex in rall:
                y_comb0 = torch.where(target == rindex, target, y_comb0)
                y_comb1 = torch.where(targetmix == rindex, targetmix, y_comb1)

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
            # it might have problems, gradients of DSC are backwards, (it would be eventually taken as BG, which is not as expected.)
            ydsclossposition = torch.zeros(y_one_hotmixup.size()[0], dtype=torch.bool)
            for kcls in range(y_one_hotmixup.size()[1]):
                ydsclosspositionccls = (y_one_hotmixup[:, kcls] > 0) & (y_one_hotmixup[:, kcls] < 1)
                ydsclossposition = ydsclossposition.float() + ydsclosspositionccls.float()
            ydsclossposition = 1 - ydsclossposition 
            ydsclossposition = ydsclossposition > 0 # this would be 0/1, 0 indicates that there are portion which I dont want.
            y_one_hotmixup = torch.floor(y_one_hotmixup)

        y_one_hotmixup = y_one_hotmixup.cuda()

        #########################################################################################################
        # p_y_given_x_train is corresponding [N,C]

        ################################### Symmetric/ asymmetric large margin loss ###################################

        '''
        I should get the input to softmax to get q, with the shape [N,C]
        '''
        if self.asy == 0:
            # have margin on all classes.
            r = [1, 1, 1]
        if self.asy == 1:
            # have margin only on foreground class
            r = [0, 1, 1]
        if self.asy == 2:
            # have margin only on the tumor class
            r = [0, 0, 1]

        # extend r from [1,C] to [N,C]
        r = torch.reshape(torch.tensor(r), [1, len(r)])
        rRepeat = torch.cat(inp.shape[0] * [r])
        # this is the input to softmax, which will give us q
        inppost = inp - rRepeat.float().cuda() * y_one_hotmixup * self.marginm

        #########################################################################################################

        # do the softmax and get q
        p_y_given_x_train = torch.softmax(inppost, 1)
        e1 = 1e-6  ## without the small margin, it would lead to nan after several epochs
        log_p_y_given_x_train = (p_y_given_x_train + e1).log()

        ################################### Symmetric/ asymmetric focal loss ###################################

        if self.asy == 0:
            # have focal reduction on all classes.
            r = [0, 0, 0]
        if self.asy == 1:
            # have focal reduction only on 0 class
            r = [0, 1, 1]
        if self.asy == 2:
            # have focal reduction only on 0 and 1 class
            r = [0, 0, 1]

        # extend r from [1,C] to [N,C]
        r = torch.reshape(torch.tensor(r), [1, len(r)])
        rRepeat = torch.cat(log_p_y_given_x_train.shape[0] * [r])

        focal_conduct_active = (1 - p_y_given_x_train + e1) ** self.gama
        focal_conduct_inactive = torch.ones(p_y_given_x_train.size())

        focal_conduct = focal_conduct_active * (1 - rRepeat.float().cuda()) + focal_conduct_inactive.cuda() * rRepeat.float().cuda()
        m_log_p_y_given_x_train = focal_conduct * log_p_y_given_x_train

        # I also need to pass the focal_conduct to the DSC loss, which is calculated outside

        #########################################################################################################

        num_samples = m_log_p_y_given_x_train.size()[0]

        loss = - (1. / num_samples) * m_log_p_y_given_x_train * y_one_hotmixup

        return loss.sum(), y_one_hotmixup, ydsclossposition, inppost, focal_conduct


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
