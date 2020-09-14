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
import torch.nn as nn


class advCrossentropyND(nn.Module):
    """
    Network has to have NO NONLINEARITY!
    """
    def __init__(self, asy):
        super(advCrossentropyND, self).__init__()
        self.asy = asy

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
        # p_y_given_x_train is corresponding [N,C]

        p_y_given_x_train = torch.softmax(inp, 1)
        e1 = 1e-6 ## without the small margin, it would lead to nan after several epochs
        log_p_y_given_x_train = (p_y_given_x_train + e1).log()

        ################################### Symmetric/ asymmetric adversarial training ###################################
        # find the directio

        if self.asy == 0:
            # have adversarial training on all classes.
            y_one_hot = y_one_hot
            ydsclosspositionc0 = (y_one_hot[:, 0] > 0)
            ydsclosspositionc1 = (y_one_hot[:, 1] > 0)
            ydsclosspositionc2 = (y_one_hot[:, 2] > 0)
            ydsclossposition = ydsclosspositionc0 + ydsclosspositionc1 + ydsclosspositionc2
        if self.asy == 1:
            # have adversarial training only on foreground
            conduct_zeros = torch.zeros(p_y_given_x_train.size()[0], 1)
            conduct_zeros = conduct_zeros.cuda()
            y_one_hot = torch.cat((conduct_zeros, y_one_hot[:, 1:3]), 1)
            ydsclosspositionc1 = (y_one_hot[:, 1] > 0)
            ydsclosspositionc2 = (y_one_hot[:, 2] > 0)
            ydsclossposition = ydsclosspositionc1 + ydsclosspositionc2
        if self.asy == 2:
            # have adversarial training only on tumor
            conduct_zeros = torch.zeros(p_y_given_x_train.size()[0], 2)
            conduct_zeros = conduct_zeros.cuda()
            y_one_hot = torch.cat((conduct_zeros, y_one_hot[:, 2:3]), 1)
            ydsclosspositionc2 = (y_one_hot[:, 2] > 0)
            ydsclossposition = ydsclosspositionc2

        #########################################################################################################

        num_samples = log_p_y_given_x_train.size()[0]

        loss = - (1. / num_samples) * log_p_y_given_x_train * y_one_hot

        # print(loss.sum())
        # print(F.cross_entropy(inp, target)) # correct and original one
        return loss.sum(), y_one_hot, ydsclossposition


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