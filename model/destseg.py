import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
from model.model_utils import ASPP, BasicBlock, l2_normalize, make_layer
from model.attcoordat import CoordAtt,senet

class TeacherNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = timm.create_model(
            "resnet18",
            pretrained=True,
            features_only=True,
            out_indices=[1, 2, 3],
        )
        # freeze teacher model
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        self.eval()
        x1, x2, x3 = self.encoder(x)
        return (x1, x2, x3)


class StudentNet(nn.Module):
    def __init__(self, ed=True):
        super().__init__()
        self.ed = ed
        self.att_coor1 = CoordAtt(256,256)
        self.att_coor2 = CoordAtt(128,128)
        self.att_coor3 = CoordAtt(64,64)
        if self.ed:
            self.decoder_layer4 = make_layer(BasicBlock, 512, 512, 2)
            self.decoder_layer3 = make_layer(BasicBlock, 512, 256, 2)
            self.decoder_layer2 = make_layer(BasicBlock, 256, 128, 2)
            self.decoder_layer1 = make_layer(BasicBlock, 128, 64, 2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.encoder = timm.create_model(
            "resnet18",
            pretrained=False,
            features_only=True,
            out_indices=[1, 2, 3, 4],
        )

    def forward(self, x,memory):
        x = self.encoder.bn1(self.encoder.conv1(x))
        x = self.encoder.maxpool(self.encoder.act1(x))
        x1 = self.encoder.layer1(x)
        x1 = memory[0](x1)["output"]
        x2 = self.encoder.layer2(x1)
        x2 = memory[1](x2)["output"]
        x3 = self.encoder.layer3(x2)
        x3 = memory[2](x3)["output"]
        x4 = self.encoder.layer4(x3)
        # x1, x2, x3, x4 = self.encoder(x)
        if not self.ed:
            return (x1, x2, x3)
        x = x4
        b4 = self.decoder_layer4(x)
        b3 = F.interpolate(b4, size=x3.size()[2:], mode="bilinear", align_corners=False)
        b3 = self.decoder_layer3(b3)
        b3 = self.att_coor1(b3)
        b2 = F.interpolate(b3, size=x2.size()[2:], mode="bilinear", align_corners=False)
        b2 = self.decoder_layer2(b2)
        b2 = self.att_coor2(b2)
        b1 = F.interpolate(b2, size=x1.size()[2:], mode="bilinear", align_corners=False)
        b1 = self.decoder_layer1(b1)
        b1 = self.att_coor3(b1)
        return {'decoder':(b1, b2, b3),'encoder':(x1,x2,x3),'latten':x4}


class SegmentationNet(nn.Module):
    def __init__(self, inplanes=448):
        super().__init__()
        self.res = make_layer(BasicBlock, inplanes, 256, 2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.se_att = senet(448)

        self.head = nn.Sequential(
            ASPP(256, 256, [6, 12, 18]),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, 1),
        )

    def forward(self, x):
        x = self.se_att(x)
        x = self.res(x)
        x = self.head(x)
        x = torch.sigmoid(x)
        return x

class MemoryUnit(nn.Module):
    def __init__(self, mem_dim, fea_dim,shrink_thres=0.0025):
        super(MemoryUnit,self).__init__()
        self.mem_dim = mem_dim
        self.fea_dim = fea_dim
        self.weight = Parameter(torch.Tensor(self.mem_dim, self.fea_dim).to(torch.device("cuda:0")))#M * C
        self.bias = None
        self.shrink_thres = shrink_thres
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv,stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv,stdv)
    
    def forward(self, input, train = False):
        if not train:
            att_weight = F.linear(input,self.weight.detach())
        else:
            att_weight = F.linear(input, self.weight)  # Fea x Mem^T, (TxC) x (CxM) = TxM
        att_weight = F.softmax(att_weight, dim=1)  # TxM
        # ReLU based shrinkage, hard shrinkage for positive value
        if(self.shrink_thres>0):
            att_weight = hard_shrink_relu(att_weight, lambd=self.shrink_thres)
            # att_weight = F.softshrink(att_weight, lambd=self.shrink_thres)
            # normalize???
            att_weight = F.normalize(att_weight, p=1, dim=1)
            # att_weight = F.softmax(att_weight, dim=1)
            # att_weight = self.hard_sparse_shrink_opt(att_weight)
        mem_trans = self.weight.permute(1, 0)  # Mem^T, MxC
        output = F.linear(att_weight, mem_trans)  # AttWeight x Mem^T^T = AW x Mem, (TxM) x (MxC) = TxC
        return {'output': output, 'att': att_weight}  # output, att_weight
    
class MemModule(nn.Module):
    def __init__(self, mem_dim, fea_dim, shrink_thres=0.0025, device='cuda'):
        super(MemModule, self).__init__()
        self.mem_dim = mem_dim #M
        self.fea_dim = fea_dim #C
        self.shrink_thres = shrink_thres
        self.memory = MemoryUnit(self.mem_dim, self.fea_dim, self.shrink_thres).to(torch.device("cuda:0"))

    def forward(self, input,is_train = False):
        s = input.data.shape
        l = len(s)

        if l == 3:
            x = input.permute(0, 2, 1)
        elif l == 4:
            x = input.permute(0, 2, 3, 1)
        elif l == 5:
            x = input.permute(0, 2, 3, 4, 1)
        else:
            x = []
            print('wrong feature map size')
        x = x.contiguous()
        x = x.view(-1, s[1])
        #
        y_and = self.memory(x,train=is_train)
        #
        y = y_and['output']
        att = y_and['att']

        if l == 3:
            y = y.view(s[0], s[2], s[1])
            y = y.permute(0, 2, 1)
            att = att.view(s[0], s[2], self.mem_dim)
            att = att.permute(0, 2, 1)
        elif l == 4:
            y = y.view(s[0], s[2], s[3], s[1])
            y = y.permute(0, 3, 1, 2)
            att = att.view(s[0], s[2], s[3], self.mem_dim)
            att = att.permute(0, 3, 1, 2)
        elif l == 5:
            y = y.view(s[0], s[2], s[3], s[4], s[1])
            y = y.permute(0, 4, 1, 2, 3)
            att = att.view(s[0], s[2], s[3], s[4], self.mem_dim)
            att = att.permute(0, 4, 1, 2, 3)
        else:
            y = x
            att = att
            print('wrong feature map size')
        return {'output': y, 'att': att}

class DeSTSeg(nn.Module):
    def __init__(self, dest=True, ed=True,mem_num=[64,128,256]):
        super().__init__()
        self.teacher_net = TeacherNet()
        self.student_net = StudentNet(ed)
        self.dest = dest
        self.segmentation_net = SegmentationNet(inplanes=448)
        self.memory = []
        for i in mem_num:
            self.memory.append(MemModule(50,fea_dim=i))

    def forward(self, img_aug, img_origin=None):
        self.teacher_net.eval()

        if img_origin is None:  # for inference
            img_origin = img_aug.clone()

        
        outputs_teacher = [
            l2_normalize(output_t) for output_t in self.teacher_net(img_origin)
        ]
        memloss = []
        i = 0
        for output_t in self.teacher_net(img_origin):
            output_m = self.memory[i](output_t,is_train=True)["output"]
            i = i + 1
            output_m = l2_normalize(output_m)
            output_t = l2_normalize(output_t.detach())
            memloss.append(1 - torch.sum(output_m * output_t, dim=1, keepdim=True))


        outputs_teacher_aug = [
            l2_normalize(output_t.detach()) for output_t in self.teacher_net(img_aug)
        ]
        outputs_student_aug = [
            l2_normalize(output_s) for output_s in self.student_net(img_aug,memory=self.memory)['decoder']
        ]
        outputs_student = [
            l2_normalize(output_s) for output_s in self.student_net(img_origin,memory=self.memory)['decoder']
        ]
        output = torch.cat(
            [
                F.interpolate(
                    output_t * output_s,
                    size=outputs_student_aug[0].size()[2:],
                    mode="bilinear",
                    align_corners=False,
                )
                for output_t, output_s in zip(outputs_teacher_aug, outputs_student_aug)
            ],
            dim=1,
        )
        # output_orgin = torch.cat(
        #     [
        #         F.interpolate(
        #             output_t * output_s,
        #             size=outputs_student_aug[0].size()[2:],
        #             mode="bilinear",
        #             align_corners=False,
        #         )
        #         for output_t, output_s in zip(outputs_teacher, outputs_student)
        #     ],
        #     dim=1,
        # )

        output_segmentation = self.segmentation_net(output)
        if self.dest:
            outputs_student = outputs_student_aug
        else:
            outputs_student = [
                l2_normalize(output_s.detach()) for output_s in self.student_net(img_origin)
            ]
        output_de_st_list = []
        for output_t, output_s in zip(outputs_teacher, outputs_student):
            a_map = 1 - torch.sum(output_s * output_t, dim=1, keepdim=True)
            output_de_st_list.append(a_map)

        return output_segmentation, output_de_st_list,memloss

def hard_shrink_relu(input, lambd=0, epsilon=1e-12):
        output = (F.relu(input-lambd) * input) / (torch.abs(input - lambd) + epsilon)
        return output
