# !/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
import imp

import __init__ as booger
import torch
import torch.nn as nn
import torch.nn.functional as F
from involution import Involution2d
import deepspeed


class ResContextBlock(nn.Module):
    def __init__(self, in_filters, out_filters, invol = False):
        super(ResContextBlock, self).__init__()
        global checkpoint
        checkpoint = deepspeed.checkpointing.checkpoint
        self.invol = invol
        self.deepspeed_checkpointing = False
        self.conv1 = nn.Conv2d(in_filters, out_filters, kernel_size=(1, 1), stride=1)
        self.act1 = nn.LeakyReLU()

        if invol:
            self.invo2 = Involution2d(out_filters, out_filters, 
            sigma_mapping = nn.Sequential(nn.LeakyReLU(), nn.BatchNorm2d(out_filters)),
            kernel_size=(3,3),stride=1, padding=(1,1), dilation=(1,1))
            self.act2 = nn.LeakyReLU()
            self.bn1 = nn.BatchNorm2d(out_filters)
        else:
            self.conv2 = nn.Conv2d(out_filters, out_filters, (3,3), padding=1)
            self.act2 = nn.LeakyReLU()
            self.bn1 = nn.BatchNorm2d(out_filters)

        if invol:
            self.invo3 = Involution2d(out_filters, out_filters, 
            sigma_mapping = nn.Sequential(nn.LeakyReLU(), nn.BatchNorm2d(out_filters)),
            kernel_size=(3,3),stride=1, padding=(2,2), dilation=(2,2))
            self.act3 = nn.LeakyReLU()
            self.bn2 = nn.BatchNorm2d(out_filters)
        else:
            self.conv3 = nn.Conv2d(out_filters, out_filters, (3,3),dilation=2, padding=2)
            self.act3 = nn.LeakyReLU()
            self.bn2 = nn.BatchNorm2d(out_filters)

    def forward(self, x):
        if self.deepspeed_checkpointing:
            shortcut = checkpoint(self.conv1, x.requires_grad_())
            shortcut = checkpoint(self.act1, shortcut.requires_grad_())

            resA = checkpoint(self.conv2, shortcut.requires_grad_())
            resA = checkpoint(self.act2, resA.requires_grad_())
            resA1 = checkpoint(self.bn1, resA.requires_grad_())

            resA = checkpoint(self.conv3, resA1.requires_grad_())
            resA = checkpoint(self.act3, resA.requires_grad_())
            resA2 = checkpoint(self.bn2, resA.requires_grad_())
        else:
            shortcut = self.conv1(x)
            shortcut = self.act1(shortcut)

            if self.invol:
                resA = self.invo2(shortcut)
                resA = self.act2(resA)
                resA1 = self.bn1(resA)
                resA = self.invo3(resA1)
                resA = self.act3(resA)
                resA2 = self.bn2(resA)
            else:
                resA = self.conv2(shortcut)
                resA = self.act2(resA)
                resA1 = self.bn1(resA)

                resA = self.conv3(resA1)
                resA = self.act3(resA)
                resA2 = self.bn2(resA)
        output = shortcut + resA2
        return output


class ResBlock(nn.Module):
    def __init__(self, in_filters, out_filters, dropout_rate, kernel_size=(3, 3), stride=1,
                 pooling=True, drop_out=True, invol=False):
        super(ResBlock, self).__init__()
        global checkpoint
        checkpoint = deepspeed.checkpointing.checkpoint
        self.invol = invol
        self.deepspeed_checkpointing = False
        self.pooling = pooling
        self.drop_out = drop_out
        self.conv1 = nn.Conv2d(in_filters, out_filters, kernel_size=(1, 1), stride=stride)
        self.act1 = nn.LeakyReLU()

        if invol:
            self.invo2 = Involution2d(in_filters,out_filters, 
            sigma_mapping = nn.Sequential(nn.LeakyReLU(), nn.BatchNorm2d(out_filters)),
            kernel_size=(3,3),stride=stride, padding=(1,1), dilation=(1,1))
        else:
            self.conv2 = nn.Conv2d(in_filters, out_filters, kernel_size=(3,3), padding=1)
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_filters)

        if invol:
            self.invo3 = Involution2d(out_filters, out_filters, 
            sigma_mapping = nn.Sequential(nn.LeakyReLU(), nn.BatchNorm2d(out_filters)),
            kernel_size=(3,3),stride=stride, padding=(2,2), dilation=(2,2))
        else:
            self.conv3 = nn.Conv2d(out_filters, out_filters, kernel_size=(3,3),dilation=2, padding=2)
        self.act3 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm2d(out_filters)
        
        if invol:
            self.invo4 = Involution2d(out_filters,out_filters, 
            sigma_mapping = nn.Sequential(nn.LeakyReLU(), nn.BatchNorm2d(out_filters)),
            kernel_size=(2,2),stride=stride, padding=(1,1), dilation=(2,2))
        else:
            self.conv4 = nn.Conv2d(out_filters, out_filters, kernel_size=(2, 2), dilation=2, padding=1)
        self.act4 = nn.LeakyReLU()
        self.bn3 = nn.BatchNorm2d(out_filters)

        self.conv5 = nn.Conv2d(out_filters*3, out_filters, kernel_size=(1, 1))
        self.act5 = nn.LeakyReLU()
        self.bn4 = nn.BatchNorm2d(out_filters)

        if pooling:
            self.dropout = nn.Dropout2d(p=dropout_rate)
            self.pool = nn.AvgPool2d(kernel_size=kernel_size, stride=2, padding=1)
        else:
            self.dropout = nn.Dropout2d(p=dropout_rate)

  
    def forward(self, x):
        if self.deepspeed_checkpointing:
            def custom():
                def custom_forward(*inputs):
                    return inputs[0]+inputs[1]
                return custom_forward
            shortcut = checkpoint(self.conv1, x.requires_grad_())
            shortcut = checkpoint(self.act1, shortcut.requires_grad_())

            resA = checkpoint(self.conv2, x.requires_grad_())
            resA = checkpoint(self.act2, resA.requires_grad_())
            resA1 = checkpoint(self.bn1, resA.requires_grad_())

            resA = checkpoint(self.conv3, resA1.requires_grad_())
            resA = checkpoint(self.act3, resA.requires_grad_())
            resA2 = checkpoint(self.bn2, resA.requires_grad_())

            resA = checkpoint(self.conv4, resA2.requires_grad_())
            resA = checkpoint(self.act4, resA.requires_grad_())
            resA3 = checkpoint(self.bn3, resA.requires_grad_())

            concat = torch.cat((resA1.requires_grad_(),resA2.requires_grad_(),resA3.requires_grad_()),dim=1)
            resA = checkpoint(self.conv5, concat)
            resA = checkpoint(self.act5, resA.requires_grad_())
            resA = checkpoint(self.bn4, resA.requires_grad_())

            resA = checkpoint(custom(), shortcut.requires_grad_() ,resA.requires_grad_())
            
        else:
            shortcut = self.conv1(x)
            shortcut = self.act1(shortcut)
            
            if self.invol:
                resA = self.invo2(x)
                resA = self.act2(resA)
                resA1 = self.bn1(resA)

                resA = self.invo3(resA1)
                resA = self.act3(resA)
                resA2 = self.bn2(resA)

                resA = self.invo4(resA2)
                resA = self.act4(resA)
                resA3 = self.bn3(resA)
            else:
                resA = self.conv2(x)
                resA = self.act2(resA)
                resA1 = self.bn1(resA)

                resA = self.conv3(resA1)
                resA = self.act3(resA)
                resA2 = self.bn2(resA)

                resA = self.conv4(resA2)
                resA = self.act4(resA)
                resA3 = self.bn3(resA)

            concat = torch.cat((resA1,resA2,resA3),dim=1)
            resA = self.conv5(concat)
            resA = self.act5(resA)
            resA = self.bn4(resA)

            resA = shortcut + resA

        if self.pooling:
            if self.drop_out:
                resB = self.dropout(resA)
            else:
                resB = resA
            resB = self.pool(resB)

            return resB, resA
        else:
            if self.drop_out:
                resB = self.dropout(resA)
            else:
                resB = resA
            return resB


class UpBlock(nn.Module):
    def __init__(self, in_filters, out_filters, dropout_rate, drop_out=True, invol=False):
        super(UpBlock, self).__init__()
        global checkpoint
        checkpoint = deepspeed.checkpointing.checkpoint
        self.invol = invol
        self.deepspeed_checkpointing = False
        self.drop_out = drop_out
        self.in_filters = in_filters
        self.out_filters = out_filters

        self.dropout1 = nn.Dropout2d(p=dropout_rate)

        self.dropout2 = nn.Dropout2d(p=dropout_rate)

        if invol:
            self.invo1 = Involution2d(in_filters//4 + 2*out_filters, out_filters, 
            sigma_mapping = nn.Sequential(nn.LeakyReLU(), nn.BatchNorm2d(out_filters)),
            kernel_size=(3,3),stride=1, padding=(1,1), dilation=(1,1))
        else:
            self.conv1 = nn.Conv2d(in_filters//4 + 2*out_filters, out_filters, (3,3), padding=1)
        self.act1 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_filters)

        if invol:
            self.invo2 = Involution2d(out_filters,out_filters, 
            sigma_mapping = nn.Sequential(nn.LeakyReLU(), nn.BatchNorm2d(out_filters)),
            kernel_size=(3,3),stride=1, padding=(2,2), dilation=(2,2))
        else:
            self.conv2 = nn.Conv2d(out_filters, out_filters, (3,3),dilation=2, padding=2)
        self.act2 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm2d(out_filters)
        
        if invol:
            self.invo3 = Involution2d(out_filters,out_filters, 
            sigma_mapping = nn.Sequential(nn.LeakyReLU(), nn.BatchNorm2d(out_filters)),
            kernel_size=(2,2),stride=1, padding=(1,1), dilation=(2,2))
        else:
            self.conv3 = nn.Conv2d(out_filters, out_filters, (2,2), dilation=2,padding=1)
        self.act3 = nn.LeakyReLU()
        self.bn3 = nn.BatchNorm2d(out_filters)

        self.conv4 = nn.Conv2d(out_filters*3,out_filters,kernel_size=(1,1))
        self.act4 = nn.LeakyReLU()
        self.bn4 = nn.BatchNorm2d(out_filters)

        self.dropout3 = nn.Dropout2d(p=dropout_rate)

    def forward(self, x, skip):
        upA = nn.PixelShuffle(2)(x)
        if self.drop_out:
            upA = self.dropout1(upA)

        upB = torch.cat((upA,skip),dim=1)
        if self.drop_out:
            upB = self.dropout2(upB)

        if self.deepspeed_checkpointing:
            upE = checkpoint(self.conv1, upB.requires_grad_())
            upE = checkpoint(self.act1, upE.requires_grad_())
            upE1 = checkpoint(self.bn1, upE.requires_grad_())

            upE = checkpoint(self.conv2, upE1.requires_grad_())
            upE = checkpoint(self.act2, upE.requires_grad_())
            upE2 = checkpoint(self.bn2, upE.requires_grad_())

            upE = checkpoint(self.conv3, upE2.requires_grad_())
            upE = checkpoint(self.act3, upE.requires_grad_())
            upE3 = checkpoint(self.bn3, upE.requires_grad_())

            concat = torch.cat((upE1.requires_grad_(),upE2.requires_grad_(),upE3.requires_grad_()),dim=1)
            upE = checkpoint(self.conv4, concat.requires_grad_())
            upE = checkpoint(self.act4, upE.requires_grad_())
            upE = checkpoint(self.bn4, upE.requires_grad_())
            upE = upE.requires_grad_()
        else:
            
            if self.invol:
                upE = self.invo1(upB)
                upE = self.act1(upE)
                upE1 = self.bn1(upE)

                upE = self.invo2(upE1)
                upE = self.act2(upE)
                upE2 = self.bn2(upE)

                upE = self.invo3(upE2)
                upE = self.act3(upE)
                upE3 = self.bn3(upE)
            else:
                upE = self.conv1(upB)
                upE = self.act1(upE)
                upE1 = self.bn1(upE)

                upE = self.conv2(upE1)
                upE = self.act2(upE)
                upE2 = self.bn2(upE)

                upE = self.conv3(upE2)
                upE = self.act3(upE)
                upE3 = self.bn3(upE)

            concat = torch.cat((upE1,upE2,upE3),dim=1)
            upE = self.conv4(concat)
            upE = self.act4(upE)
            upE = self.bn4(upE)
        if self.drop_out:
            upE = self.dropout3(upE)

        return upE


class SalsaNext(nn.Module):
    def __init__(self, nclasses):
        super(SalsaNext, self).__init__()
        global checkpoint
        checkpoint = deepspeed.checkpointing.checkpoint
        self.nclasses = nclasses
        self.deepspeed_checkpointing = True

        self.downCntx = ResContextBlock(5, 32, invol=False)
        self.downCntx2 = ResContextBlock(32, 32, invol=False)
        self.downCntx3 = ResContextBlock(32, 32, invol=True)

        self.resBlock1 = ResBlock(32, 2 * 32, 0.2, pooling=True, drop_out=False, invol=False)
        self.resBlock2 = ResBlock(2 * 32, 2 * 2 * 32, 0.2, pooling=True, invol=False)
        self.resBlock3 = ResBlock(2 * 2 * 32, 2 * 4 * 32, 0.2, pooling=True, invol=False)
        self.resBlock4 = ResBlock(2 * 4 * 32, 2 * 4 * 32, 0.2, pooling=True, invol=False)
        self.resBlock5 = ResBlock(2 * 4 * 32, 2 * 4 * 32, 0.2, pooling=False, invol=False)

        self.upBlock1 = UpBlock(2 * 4 * 32, 4 * 32, 0.2, invol=False)
        self.upBlock2 = UpBlock(4 * 32, 4 * 32, 0.2, invol=False)
        self.upBlock3 = UpBlock(4 * 32, 2 * 32, 0.2, invol=False)
        self.upBlock4 = UpBlock(2 * 32, 32, 0.2, drop_out=False, invol=False)

        self.logits = nn.Conv2d(32, nclasses, kernel_size=(1, 1))

        if self.deepspeed_checkpointing:
            self.layers = [self.downCntx, self.downCntx2, self.downCntx3, 
        self.resBlock1,self.resBlock2,self.resBlock3, self.resBlock4, self.resBlock5,
        self.upBlock1, self.upBlock2, self.upBlock3, self.upBlock4]

    def checkpoint_forward(self, x, chunk_length=1):
        def custom(start, end):
            def custom_forward(*inputs):
                x_ = inputs[0]
                new_y_array = [[0]]
                for ind, layer in enumerate(self.layers):
                    if start <= ind < end:
                        if 0<= ind < 3:
                            x_ = layer(x_)
                        elif 3<=ind < 7:
                            x_, y_t = layer(x_)
                            new_y_array.append(y_t)
                        elif ind == 7:
                            x_ = layer(x_)

                return (x_, *new_y_array)
            return custom_forward
        def custom_decoder():
            def custom_forward_decoder(*inputs):
                x_ = inputs[0]
                new_y_array = inputs[1:]
                layers = self.layers[8:12]
                for layer in layers:
                    x_ = layer(x_, new_y_array[-1])
                    new_y_array=new_y_array[:-1]
                return x_
            return custom_forward_decoder

        l = 0
        num_layers = 8
        hidden_states = x
        
        y_array = []
        tmp = torch.ones(1,dtype=torch.float32, requires_grad=True)
        while l < num_layers:
            end = l+chunk_length
            if end > num_layers:
                end = num_layers
            final = checkpoint(custom(l, end), hidden_states.requires_grad_())
            if len(final)==1:
                hidden_states = final[0]
            else:
                hidden_states = final[0]
                y_array_elems = final[2:]
                y_array = [*y_array, *y_array_elems]

            l = end
        for ind, elem in enumerate(y_array):
            y_array[ind]=elem.requires_grad_()
        hidden_states = checkpoint(custom_decoder(), hidden_states.requires_grad_(), *y_array)
        return hidden_states

    def forward(self, x):
        if self.deepspeed_checkpointing:
            if False:
                up1e = self.checkpoint_forward(x)

                up1e = up1e.requires_grad_()
            else:
                downCntx = checkpoint(self.downCntx, x.requires_grad_())
                downCntx = checkpoint(self.downCntx2,downCntx)
                downCntx = checkpoint(self.downCntx3,downCntx)
                down0c, down0b = checkpoint(self.resBlock1,downCntx)
                down1c, down1b = checkpoint(self.resBlock2,down0c)
                down2c, down2b = checkpoint(self.resBlock3,down1c)
                down3c, down3b = checkpoint(self.resBlock4,down2c)
                down5c = checkpoint(self.resBlock5,down3c)

                up4e = checkpoint(self.upBlock1,down5c,down3b)
                up3e = checkpoint(self.upBlock2,up4e, down2b)
                up2e = checkpoint(self.upBlock3,up3e, down1b)
                up1e = checkpoint(self.upBlock4,up2e, down0b)
                up1e = up1e.requires_grad_()
        else:
            downCntx = self.downCntx(x)
            downCntx = self.downCntx2(downCntx)
            downCntx = self.downCntx3(downCntx)
            down0c, down0b = self.resBlock1(downCntx)
            down1c, down1b = self.resBlock2(down0c)
            down2c, down2b = self.resBlock3(down1c)
            down3c, down3b = self.resBlock4(down2c)
            down5c = self.resBlock5(down3c)

            up4e = self.upBlock1(down5c,down3b)
            up3e = self.upBlock2(up4e, down2b)
            up2e = self.upBlock3(up3e, down1b)
            up1e = self.upBlock4(up2e, down0b)
        logits = self.logits(up1e)

        logits = F.softmax(logits, dim=1)

        return logits