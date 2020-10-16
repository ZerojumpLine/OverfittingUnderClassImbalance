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

class mCrossentropyND(nn.Module):
    """
    Network has to have NO NONLINEARITY!
    """
    def __init__(self, asy, gama, margin, weights):
        super(mCrossentropyND, self).__init__()
        self.asy = asy
        self.gama = gama
        self.margin = margin
        self.weights = weights

    def forward(self, inp, target):
        target = target.long()
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

        # now inp is [N,C], target is [N,]

        y_one_hot = self.one_hot_embedding(target.data.cpu(), num_classes)
        y_one_hot = y_one_hot.cuda()
        y_one_hot[:,2] = y_one_hot[:,2] * self.weights
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
        inppost = inp - rRepeat.float().cuda() * y_one_hot * self.margin

        #########################################################################################################

        # do the softmax and get q
        p_y_given_x_train = torch.softmax(inppost, 1)
        e1 = 1e-6  ## without the small margin, it would lead to nan after several epochs
        log_p_y_given_x_train = (p_y_given_x_train + e1).log()

        ################################### Symmetric/ asymmetric focal loss ###################################

        if self.asy == 0:
            # have focal reduction on all classes.
            r = [1, 1, 1]
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

        focal_conduct = focal_conduct_active * (1-rRepeat.float().cuda()) + focal_conduct_inactive.cuda() * rRepeat.float().cuda()
        m_log_p_y_given_x_train = focal_conduct * log_p_y_given_x_train

        # I also need to pass the focal_conduct to the DSC loss, which is calculated outside

        #########################################################################################################

        num_samples = m_log_p_y_given_x_train.size()[0]

        loss = - (1. / num_samples) * m_log_p_y_given_x_train * y_one_hot
        # print(loss.sum())
        return loss.sum(), inppost, focal_conduct


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