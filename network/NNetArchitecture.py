import torch.nn.functional as F
import torch.nn as nn
import torch
import sys

sys.path.append('..')

 
class NNetArchitecture(nn.Module):
    def __init__(self, game, args):
        super(NNetArchitecture, self).__init__()
        # game params
        self.feat_cnt = args.feat_cnt
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args
        self.conv1 = nn.Conv2d(3, 1, 5)
        self.conv2 = nn.Conv1d(5, 16,3)
        self.fc1 = nn.Linear(3*9*9, 120)
        self.fc2 = nn.Linear(120, 256)
        self.fc3 = nn.Linear(256,self.action_size)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
        # 也可以判断是否为conv2d，使用相应的初始化方式 
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, s):
        # batch_size x feat_cnt x board_x x board_y
        s = s.view(-1, self.feat_cnt, self.board_x, self.board_y)   
        s = s.reshape(s.size(0), 243)
        """
            Design your neural network architecture
            Return a probability distribution of the next play (an array of length self.action_size) 
            and the evaluation of the current state.
        """
        # out = F.relu(self.conv1(s))
        # out = F.max_pool2d(out, 2)
        # out = F.relu(self.conv2(out))
        # out = F.max_pool2d(out, 2)
        # out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(s))
        out = F.relu(self.fc2(out))
        pi = self.fc3(out)
        # Think: What are the advantages of using log_softmax ?
        v = torch.tensor([0])
        return F.log_softmax(pi, dim=1), torch.tanh(v)