import torch.nn as nn
import torch


class GCN(nn.Module):
    def __init__(self, num_state, num_node):
        super(GCN, self).__init__()
        self.num_state = num_state
        self.num_node = num_node
        self.conv1 = nn.Conv1d(num_node, num_node, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1)

    def forward(self, seg, aj):
        n, c, h, w = seg.size()
        seg = seg.view(n, self.num_state, -1).contiguous()
        seg_similar = torch.bmm(seg, aj)
        out = self.relu(self.conv2(seg_similar))
        output = out + seg
       
        return output


class EAGCN(nn.Module):
    def __init__(self, num_in, plane_mid, mids, normalize=False):  
        super(EAGCN, self).__init__()
        self.num_in = num_in
        self.mids = mids
        self.normalize = normalize
        self.num_s = int(plane_mid)
        self.num_n = (mids) * (mids)
        self.maxpool_c = nn.AdaptiveAvgPool2d(output_size=(1))
        self.conv_s1 = nn.Conv2d(num_in, self.num_s, kernel_size=1)
        self.conv_s11 = nn.Conv2d(num_in, self.num_s, kernel_size=1)
        self.conv_s2 = nn.Conv2d(1, 1, kernel_size=1)
        self.conv_s3 = nn.Conv2d(1, 1, kernel_size=1)
        self.mlp = nn.Linear(num_in, self.num_s)
        self.fc = nn.Conv2d(num_in, self.num_s, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.downsample = nn.AdaptiveAvgPool2d(output_size=(mids, mids))


        self.gcn = GCN(num_state=num_in, num_node=self.num_n)
        self.conv_extend = nn.Conv2d(self.num_s, num_in, kernel_size=1)
        self.blocker = nn.BatchNorm2d(num_in)

    def forward(self, seg_ori, edge_ori): 


        seg = seg_ori 
        edge = edge_ori 
        n, c, h, w = seg.size()

        
        seg_s = self.conv_s1(seg)
        theta_T = seg_s.view(n, self.num_s, -1).contiguous() 
        theta = seg_s.view(n, -1, self.num_s).contiguous() 
        channel_att = torch.relu(self.mlp(self.maxpool_c(seg).squeeze(3).squeeze(2))).view(n, self.num_s, -1)  
        diag_channel_att = torch.bmm(channel_att, channel_att.view(n, -1, self.num_s)) 

        similarity_c = torch.bmm(theta, diag_channel_att) 
        similarity_c = self.softmax(torch.bmm(similarity_c, theta_T))  
        

        seg_c = self.conv_s11(seg)
        sigma = seg_c.view(n, self.num_s, -1).contiguous()
        sigma_T = seg_c.view(n, -1, self.num_s).contiguous()
        sigma_out = torch.bmm(sigma_T, sigma)

        edge_m = seg * edge

        maxpool_s, _ = torch.max(seg, dim=1)
        edge_m_pool, _ = torch.max(edge_m, dim=1)

        seg_ss = self.conv_s2(maxpool_s.unsqueeze(1)).view(n, 1, -1)  
        edge_mm = self.conv_s3(edge_m_pool.unsqueeze(1)).view(n, -1, 1)

        diag_spatial_att = torch.bmm(edge_mm, seg_ss) * sigma_out
        similarity_s = self.softmax(diag_spatial_att)
        similarity = similarity_c + similarity_s


        seg_gcn = self.gcn(seg, similarity).view(n, self.num_in, self.mids, self.mids)

        ext_up_seg_gcn = seg_gcn + seg_ori
        return ext_up_seg_gcn


class GRU_EAGCN(nn.Module):
    def __init__(self, num_in, plane_mid, mids):
        super(GRU_EAGCN, self).__init__()

        self.eagcn = EAGCN(num_in, plane_mid, mids)
        self.rnn = torch.nn.GRU(input_size=1024, hidden_size=1024, num_layers=1)
    def forward(self, seg, edge):
        _, c, h, w = seg.size()
        #------------t0-------#
        updated_seg = self.eagcn(seg, edge)
        updated_seg = updated_seg.view(c, -1, h*w)
        output_0, h_0 = self.rnn(updated_seg)
        # ------------t1-------#
        output = output_0.view(-1, c, h, w)
        updated_seg = self.eagcn(output, edge)
        updated_seg = updated_seg.view(c, -1, h*w)
        output_1, h_1 = self.rnn(updated_seg, h_0)
        #-------------t2----------#
        output = output_1.view(-1, c, h, w)
        updated_seg = self.eagcn(output, edge)
        updated_seg = updated_seg.view(c, -1, h*w)
        output_2, h_2 = self.rnn(updated_seg, h_1)

        # reshape back
        output_2 = output_2.view(-1, c, h, w)

        return output_2







