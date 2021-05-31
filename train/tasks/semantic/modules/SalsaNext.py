# !/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
import imp

import __init__ as booger
import torch
import torch.nn as nn
import torch.nn.functional as F

import deepspeed


class ResContextBlock(nn.Module):
    def __init__(self, in_filters, out_filters):
        super(ResContextBlock, self).__init__()
        global checkpoint
        checkpoint = deepspeed.checkpointing.checkpoint
        self.deepspeed_checkpointing = False
        self.conv1 = nn.Conv2d(in_filters, out_filters, kernel_size=(1, 1), stride=1)
        self.act1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(out_filters, out_filters, (3,3), padding=1)
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_filters)

        self.conv3 = nn.Conv2d(out_filters, out_filters, (3,3),dilation=2, padding=2)
        self.act3 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm2d(out_filters)

        if self.deepspeed_checkpointing:
            self.layers = [self.conv1, self.act1,
            self.conv2, self.act2, self.bn1,
            self.conv3, self.act3, self.bn2]

    # Forwarding using deepspeed checkpoints
    def checkpoint_forward(self, x, chunk_length=1):
        def custom(start, end):
            def custom_forward(*inputs):
                x_ = inputs[0]
                shortcut_array = []
                for ind, layer in enumerate(self.layers):
                    if start <= ind < end:
                        x_ = layer(x_)
                        if ind == 1:
                            shortcut_array.append(x_)
                      
                return x_, shortcut_array
            return custom_forward

        l = 0
        num_layers = len(self.layers)
        hidden_states = x
        shortcut_array = []
        while l < num_layers:
            end = l+chunk_length
            if end > num_layers:
                end = num_layers
            if len(shortcut_array) > 0:
                hidden_states, _ = checkpoint(custom(l, end),hidden_states)
            else:
                hidden_states, shortcut_array = checkpoint(custom(l, end),hidden_states)
            l = end
        return hidden_states, shortcut_array[0]

    def forward(self, x):
        if self.deepspeed_checkpointing:
            resA2, shortcut = self.checkpoint_forward(x)
        else:
            shortcut = self.conv1(x)
            shortcut = self.act1(shortcut)

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
                 pooling=True, drop_out=True):
        super(ResBlock, self).__init__()
        global checkpoint
        checkpoint = deepspeed.checkpointing.checkpoint
        self.deepspeed_checkpointing = False
        self.pooling = pooling
        self.drop_out = drop_out
        self.conv1 = nn.Conv2d(in_filters, out_filters, kernel_size=(1, 1), stride=stride)
        self.act1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(in_filters, out_filters, kernel_size=(3,3), padding=1)
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_filters)

        self.conv3 = nn.Conv2d(out_filters, out_filters, kernel_size=(3,3),dilation=2, padding=2)
        self.act3 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm2d(out_filters)

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
        
        if self.deepspeed_checkpointing:
            self.layers = [self.conv1, self.act1,
        self.conv2, self.act2, self.bn1,
        self.conv3, self.act3, self.bn2,
        self.conv4, self.act4, self.bn3,
        self.conv5, self.act5, self.bn4]

    # Forwarding using deepspeed checkpoints
    def checkpoint_forward(self, x, chunk_length=1):
        def custom(start, end):
            def custom_forward(*inputs):
                x_ = inputs[0][0]
                accum_array = inputs[0][1]
                shortcut_array = []
                for ind, layer in enumerate(self.layers):
                    if start <= ind < end:
                        if ind == 11:
                            x_ = torch.cat(accum_array,dim=1)
                            accum_array = []
                        if ind == 2:
                            og_input = inputs[0][2]
                            x_ = layer(og_input)
                        else:
                            x_ = layer(x_)
                        if ind == 1:
                            shortcut_array.append(x_)
                        if ind == 4 or ind == 7 or ind == 10:
                            accum_array.append(x_)
                      
                return x_, accum_array, shortcut_array
            return custom_forward

        l = 0
        num_layers = len(self.layers)
        hidden_states = x
        accum_array = []
        shortcut_array = []
        while l < num_layers:
            end = l+chunk_length
            if end > num_layers:
                end = num_layers
            if len(shortcut_array)>0:
                hidden_states, accum_array, _ = checkpoint(custom(l, end),[hidden_states, accum_array, x])
            else:
                hidden_states, accum_array, shortcut_array = checkpoint(custom(l, end),[hidden_states, accum_array, x])
            l = end
        return hidden_states, shortcut_array[0]

    def forward(self, x):
        if self.deepspeed_checkpointing:
            resA, shortcut = self.checkpoint_forward(x)
        else:
            shortcut = self.conv1(x)
            shortcut = self.act1(shortcut)

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
    def __init__(self, in_filters, out_filters, dropout_rate, drop_out=True):
        super(UpBlock, self).__init__()
        global checkpoint
        checkpoint = deepspeed.checkpointing.checkpoint
        self.deepspeed_checkpointing = False
        self.drop_out = drop_out
        self.in_filters = in_filters
        self.out_filters = out_filters

        self.dropout1 = nn.Dropout2d(p=dropout_rate)

        self.dropout2 = nn.Dropout2d(p=dropout_rate)

        self.conv1 = nn.Conv2d(in_filters//4 + 2*out_filters, out_filters, (3,3), padding=1)
        self.act1 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_filters)

        self.conv2 = nn.Conv2d(out_filters, out_filters, (3,3),dilation=2, padding=2)
        self.act2 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm2d(out_filters)

        self.conv3 = nn.Conv2d(out_filters, out_filters, (2,2), dilation=2,padding=1)
        self.act3 = nn.LeakyReLU()
        self.bn3 = nn.BatchNorm2d(out_filters)


        self.conv4 = nn.Conv2d(out_filters*3,out_filters,kernel_size=(1,1))
        self.act4 = nn.LeakyReLU()
        self.bn4 = nn.BatchNorm2d(out_filters)

        self.dropout3 = nn.Dropout2d(p=dropout_rate)

        if self.deepspeed_checkpointing:
            self.layers = [self.conv1, self.act1, self.bn1,
        self.conv2, self.act2, self.bn2,
        self.conv3, self.act3, self.bn3,
        self.conv4, self.act4, self.bn4]

    def checkpoint_forward(self, x, chunk_length=1):
        def custom(start, end):
            def custom_forward(*inputs):
                x_ = inputs[0][0]
                accum_array = inputs[0][1]
                for ind, layer in enumerate(self.layers):
                    if start<=ind<end:
                        if ind == 9:
                            x_ = torch.cat(accum_array,dim=1)
                            accum_array = []
                        x_ = layer(x_)
                        if ind == 2 or ind == 5 or ind == 8:
                            accum_array.append(x_)
                      
                return x_, accum_array
            return custom_forward

        l = 0
        num_layers = len(self.layers)
        chunk_length = 1
        hidden_states = x
        accum_array = []
        while l < num_layers:
            end = l+chunk_length
            if end > num_layers:
                end = num_layers
            hidden_states, accum_array = checkpoint(custom(l, end),[hidden_states, accum_array])
            l = end

    def forward(self, x, skip):
        upA = nn.PixelShuffle(2)(x)
        if self.drop_out:
            upA = self.dropout1(upA)

        upB = torch.cat((upA,skip),dim=1)
        if self.drop_out:
            upB = self.dropout2(upB)

        if self.deepspeed_checkpointing:
            upE = self.checkpoint_forward(upB)
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

        self.downCntx = ResContextBlock(5, 32)
        self.downCntx2 = ResContextBlock(32, 32)
        self.downCntx3 = ResContextBlock(32, 32)

        self.resBlock1 = ResBlock(32, 2 * 32, 0.2, pooling=True, drop_out=False)
        self.resBlock2 = ResBlock(2 * 32, 2 * 2 * 32, 0.2, pooling=True)
        self.resBlock3 = ResBlock(2 * 2 * 32, 2 * 4 * 32, 0.2, pooling=True)
        self.resBlock4 = ResBlock(2 * 4 * 32, 2 * 4 * 32, 0.2, pooling=True)
        self.resBlock5 = ResBlock(2 * 4 * 32, 2 * 4 * 32, 0.2, pooling=False)

        self.upBlock1 = UpBlock(2 * 4 * 32, 4 * 32, 0.2)
        self.upBlock2 = UpBlock(4 * 32, 4 * 32, 0.2)
        self.upBlock3 = UpBlock(4 * 32, 2 * 32, 0.2)
        self.upBlock4 = UpBlock(2 * 32, 32, 0.2, drop_out=False)

        self.logits = nn.Conv2d(32, nclasses, kernel_size=(1, 1))

        if self.deepspeed_checkpointing:
            self.layers = [self.downCntx, self.downCntx2, self.downCntx3, 
        self.resBlock1,self.resBlock2,self.resBlock3, self.resBlock4, self.resBlock5,
        self.upBlock1, self.upBlock2, self.upBlock3, self.upBlock4]

    def checkpoint_forward(self, x, chunk_length=3):
        def custom(start, end):
            def custom_forward(*inputs):
                x_ = inputs[0][0]
                y_array = inputs[0][1]
                for ind, layer in enumerate(self.layers):
                    if start <= ind < end:
                        if ind < 3:
                            x_ = layer(x_)
                        elif ind < 7:
                            x_, y_t = layer(x_)
                            y_array.append(y_t)
                        elif ind == 7:
                            x_ = layer(x_)
                        elif ind < 12:
                            x_ = layer(x_, y_array[-1])
                            y_array=y_array[:-1]
                return x_, y_array
            return custom_forward

        l = 0
        num_layers = len(self.layers)
        hidden_states = x
        y_array = []
        while l < num_layers:
            end = l+chunk_length
            if end > num_layers:
                end = num_layers
            hidden_states, y_array = checkpoint(custom(l, end), [hidden_states, y_array])
            l = end
        return hidden_states

    def forward(self, x):
        if self.deepspeed_checkpointing:
            up1e = self.checkpoint_forward(x)
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