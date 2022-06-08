import torch
from torch.nn import Module, Conv2d, Linear, ReLU, MaxPool2d, LayerNorm, Sequential, ConvTranspose2d, Sigmoid


# dw_conv k*k
class DWConv(Module):
    def __init__(self, input_dim, output_dim, k: int = 3):
        super(DWConv, self).__init__()
        self.depth_wise = Conv2d(input_dim, input_dim, kernel_size=(k, k), groups=input_dim, padding=(1, 1))
        self.point_wise = Conv2d(input_dim, output_dim, kernel_size=(1, 1))

    def forward(self, x):
        out = self.depth_wise(x)
        out = self.point_wise(out)
        return out


class AC(Module):
    def __init__(self, input_dim, resnet: bool = False, ratio: float = 4):
        super(AC, self).__init__()
        self.s1 = Sequential(
            Linear(input_dim, int(input_dim / ratio)),
            ReLU(inplace=True),
            Linear(int(input_dim / ratio), input_dim),
            Sigmoid()
        )
        self.f = resnet
        self.i_d = input_dim

    def forward(self, x):
        x1 = torch.mean(x, dim=(2, 3))
        if self.f is False:
            y = self.s1(x1).view(-1, self.i_d, 1, 1) * x
        else:
            y = self.s1(x1).view(-1, self.i_d, 1, 1) * x + x
        return y


class ODAS(Module):
    def __init__(self):
        super(ODAS, self).__init__()
        self.p0 = Conv2d(1, 16, kernel_size=(7, 7), stride=(1, 1))
        self.p1 = self.part_1()
        self.max_pool1, self.s1, self.conv1_o, self.s2, self.s3, self.max_pool2, self.conv1_1, self.s4, self.s5, self.s6 = self.part_2()
        self.s7, self.s8, self.s9, self.s10, self.s11, self.resize_1 = self.part_3()
        self.conv1_2, self.s12, self.s13, self.ac_1, self.s14, self.s15, self.resize_2 = self.part_4()

    def forward(self, x):
        y = self.p0(x)
        y1 = self.p1(y)

        t1 = self.max_pool1(y)

        y2_1_1 = self.s1(t1)
        y2_1_2 = self.conv1_o(t1)
        y2_1 = y2_1_1 + y2_1_2

        y2_2 = self.s2(y2_1) + y2_1

        y2_3_1 = self.s3(y2_2)
        y2_3_2 = self.max_pool2(y2_2)
        y2_3 = y2_3_1 + y2_3_2

        y2_4 = self.conv1_1(y2_3)

        y2_5 = self.s4(y2_4) + y2_4

        y2_6 = self.s5(y2_5) + y2_5

        y2 = self.s6(y2_6)

        y3_1 = y1 + y2

        y3_2 = self.s7(y3_1) + y3_1
        y3_3 = self.s8(y3_2) + y3_2
        y3_4 = self.s9(y3_3) + y3_3
        y3_5 = self.s10(y3_4) + y3_4
        y3 = self.resize_1(self.s11(y3_5) + y3_5)

        y4_1 = self.conv1_2(t1)
        y4_2 = self.s12(y4_1) + y4_1
        y4_3 = self.s13(y4_2) + y4_2
        y4_4 = self.ac_1(y4_3)

        y4_5 = y4_4 + y3
        y4_6 = self.s14(y4_5) + y4_5
        y4_7 = self.s15(y4_6) + y4_6
        y4 = torch.softmax(self.resize_2(y4_7), dim=1)

        return y4

    @staticmethod
    def part_1():
        ac_1 = AC(16, True)
        conv1_1 = Conv2d(16, 32, kernel_size=(1, 1), stride=(1, 1))
        dw_conv_3_1 = DWConv(32, 32)
        conv1_2 = Conv2d(32, 128, kernel_size=(1, 1), stride=(1, 1))

        resize1 = Conv2d(128, 256, kernel_size=(5, 5), stride=(3, 3))
        return Sequential(
            ac_1,
            conv1_1,
            dw_conv_3_1,
            conv1_2,
            resize1
        )

    @staticmethod
    def part_2():
        max_pool1 = MaxPool2d(kernel_size=(3, 3), stride=(3, 3))

        conv1_1 = Conv2d(16, 32, kernel_size=(1, 1), stride=(1, 1))
        ac_1 = AC(32, True)
        conv1_2 = Conv2d(32, 128, kernel_size=(1, 1), stride=(1, 1))
        s1 = Sequential(
            conv1_1,
            LayerNorm([168, 168]),
            ReLU(inplace=True),
            ac_1,
            conv1_2,
            LayerNorm([168, 168]),
            ReLU(inplace=True)
        )
        conv1_o = Conv2d(16, 128, kernel_size=(1, 1), stride=(1, 1))

        conv1_3 = Conv2d(128, 32, kernel_size=(1, 1), stride=(1, 1))
        ac_2 = AC(32, True)
        conv1_4 = Conv2d(32, 128, kernel_size=(1, 1), stride=(1, 1))
        s2 = Sequential(
            conv1_3,
            LayerNorm([168, 168]),
            ReLU(inplace=True),
            ac_2,
            conv1_4,
            LayerNorm([168, 168]),
            ReLU(inplace=True),
        )

        conv1_5 = Conv2d(128, 32, kernel_size=(1, 1), stride=(1, 1))
        dw_conv_3_1 = DWConv(32, 64)
        conv1_6 = Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1))
        conv1_7 = Conv2d(32, 128, kernel_size=(1, 1), stride=(1, 1))
        s3 = Sequential(
            conv1_5,
            LayerNorm([168, 168]),
            ReLU(inplace=True),
            dw_conv_3_1,
            conv1_6,
            LayerNorm([168, 168]),
            ReLU(inplace=True),
            conv1_7,
            LayerNorm([168, 168]),
            ReLU(inplace=True),
        )
        max_pool2 = MaxPool2d(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        conv1_8_0 = Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1))
        conv1_8_1 = Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
        conv1_8 = Sequential(
            conv1_8_0,
            LayerNorm([168, 168]),
            ReLU(inplace=True),
            conv1_8_1,
            LayerNorm([168, 168]),
            ReLU(inplace=True),
        )

        dw_conv_3_2 = DWConv(128, 512)
        conv1_9 = Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1))
        s4 = Sequential(
            dw_conv_3_2,
            conv1_9,
            LayerNorm([168, 168]),
            ReLU(inplace=True),
        )

        dw_conv_3_3 = DWConv(128, 512)
        conv1_10 = Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1))
        s5 = Sequential(
            dw_conv_3_3,
            conv1_10,
            LayerNorm([168, 168]),
            ReLU(inplace=True),
        )

        dw_conv_3_4 = DWConv(128, 256)
        conv1_11 = Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
        s6 = Sequential(
            dw_conv_3_4,
            conv1_11,
            LayerNorm([168, 168]),
            ReLU(inplace=True),
        )

        return max_pool1, s1, conv1_o, s2, s3, max_pool2, conv1_8, s4, s5, s6

    @staticmethod
    def part_3():
        # 168
        max_pool_1 = MaxPool2d(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # 168
        dw_conv_3_1 = DWConv(256, 256)
        # 168
        conv1_1 = Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
        # 168
        s1 = Sequential(
            max_pool_1,
            dw_conv_3_1,
            conv1_1,
            LayerNorm([168, 168]),
            ReLU(inplace=True),
        )

        max_pool_2 = MaxPool2d(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        dw_conv_3_2 = DWConv(256, 256)
        conv1_2 = Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
        s2 = Sequential(
            max_pool_2,
            dw_conv_3_2,
            conv1_2,
            LayerNorm([168, 168]),
            ReLU(inplace=True),
        )

        dw_conv_3_3 = DWConv(256, 512)
        conv1_3 = Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1))
        conv1_3_1 = Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1))
        s3 = Sequential(
            dw_conv_3_3,
            conv1_3,
            LayerNorm([168, 168]),
            ReLU(inplace=True),
            conv1_3_1,
            LayerNorm([168, 168]),
            ReLU(inplace=True),
        )

        dw_conv_3_4 = DWConv(256, 128)
        conv1_4 = Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1))
        s4 = Sequential(
            dw_conv_3_4,
            conv1_4,
            LayerNorm([168, 168]),
            ReLU(inplace=True),
        )

        dw_conv_3_5 = DWConv(256, 256)
        conv1_5 = Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
        s5 = Sequential(
            dw_conv_3_5,
            conv1_5,
            LayerNorm([168, 168]),
            ReLU(inplace=True),
        )

        resize = Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))

        return s1, s2, s3, s4, s5, resize

    @staticmethod
    def part_4():
        conv1_1 = Conv2d(16, 128, kernel_size=(1, 1), stride=(1, 1))

        ac_1 = AC(128, True, ratio=2)
        ac_2 = AC(128, True)
        s1 = Sequential(
            ac_1,
            ac_2
        )

        ac_3 = AC(128, True, ratio=2)
        ac_4 = AC(128, True)
        s2 = Sequential(
            ac_3,
            ac_4
        )

        ac_5 = AC(128, ratio=0.5)

        max_pool_1 = MaxPool2d(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        ac_6 = AC(128, True, ratio=1)
        s3 = Sequential(
            max_pool_1,
            ac_6
        )

        max_pool_2 = MaxPool2d(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        ac_7 = AC(128, True, ratio=1)
        s4 = Sequential(
            max_pool_2,
            ac_7
        )

        resize_1 = ConvTranspose2d(128, 2, kernel_size=(11, 11), stride=(3, 3))

        return conv1_1, s1, s2, ac_5, s3, s4, resize_1
