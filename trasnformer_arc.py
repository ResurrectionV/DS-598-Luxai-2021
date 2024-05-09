class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=1,kernel_size=(4,4),stride=1)
        self.conv2 = nn.Conv2d(in_channels=2,out_channels=1,kernel_size=(4,4),stride=1)
        self.conv3 = nn.Conv2d(in_channels=2,out_channels=1,kernel_size=(4,4),stride=1)
        self.conv4 = nn.Conv2d(in_channels=9,out_channels=1,kernel_size=(4,4),stride=1)
        self.lstm = nn.LSTM(input_size=676, hidden_size=128, num_layers=2, batch_first=True)
        self.relu = nn.ReLU()
        self.emb = nn.Embedding(4,4)
        self.encoders = nn.TransformerEncoderLayer(d_model=260,nhead=10)
        self.enc = nn.TransformerEncoder(self.encoders,num_layers=3)
        self.dec = nn.Linear(260,config.num_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=config.lr, weight_decay=config.weight_decay)
#         self.initialize_hidden()

        self.to(config.device)

    def initialize_hidden(self):
        self.h = torch.zeros([2, 1, 128], requires_grad=True).to(config.device)
        self.c = torch.zeros([2, 1, 128], requires_grad=True).to(config.device)


    def forward(self, state1, state2):
        # Ensure all inputs are moved to the same device
        state1 = [component.to(config.device) for component in state1]
        state2 = [component.to(config.device) for component in state2]

        [r_map1, c_map1, u_map1, e_types, e_info], [r_map2, c_map2, u_map2] = state1, state2
        for idx,component in enumerate([r_map1, c_map1, u_map1, e_types, e_info, r_map2, c_map2, u_map2]):
            component.to(config.device)

        outr1 = self.relu(self.conv1(r_map1))
        outc1 = self.relu(self.conv2(c_map1)).permute(1,0,2,3)
        outu1 = self.relu(self.conv3(u_map1)).permute(1,0,2,3)
        map1 = torch.cat([outr1, outc1, outu1],dim=1)
        map1 = self.relu(self.conv4(map1))

        outr2 = self.relu(self.conv1(r_map2))
        outc2 = self.relu(self.conv2(c_map2)).permute(1,0,2,3)
        outu2 = self.relu(self.conv3(u_map2)).permute(1,0,2,3)
        map2 = torch.cat([outr2, outc2, outu2],dim=1)
        map2 = self.relu(self.conv4(map2))

        map1 = map1.view(1,1,-1)
        map2 = map2.view(1,1,-1)
        map_ = torch.cat([map1, map2],dim=1)

        map_, _ = self.lstm(map_)
        map_ = map_.view(1,-1).expand(1,len(e_types),-1)

        e_embs = self.emb(e_types)
        embs = torch.add(e_embs, e_info).unsqueeze(0)

        out = torch.cat([embs, map_],dim=2)
        out = self.relu(self.enc(out))
        out = self.dec(out)

        invalid_actions = torch.zeros(out.shape)

        invalid_actions[0,e_types==UNIT_TYPE['WORKER'],:2] = -1e30
        invalid_actions[0,e_types==UNIT_TYPE['CART'],:2] = -1e30
        invalid_actions[0,e_types==2,2:] = -1e30

        out = out.to(config.device)
        invalid_actions = invalid_actions.to(config.device)
        out = torch.add(out,invalid_actions)
        if config.debug_print:
            print(f'preds\n{out}')
        out = torch.softmax(out,dim=2).squeeze(0)

        return out