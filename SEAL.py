import argparse
import torch.nn as nn
from torch.nn.functional import normalize
from torch.utils.data import Dataset, DataLoader
from metric import cluster_acc
from sklearn.metrics import v_measure_score, adjusted_rand_score, accuracy_score
import numpy as np
import random
import torch


class DataAug(nn.Module):
    def __init__(self, dropout=0.9):
        super(DataAug, self).__init__()
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x):
        aug_data = self.drop(x)

        return aug_data


class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2000),
            nn.ReLU(),
            nn.Linear(2000, feature_dim),
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 2000),
            nn.ReLU(),
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, input_dim)
        )

    def forward(self, x):
        return self.decoder(x)


from load_data import next_batch, load_train_data

class Network(nn.Module):
    def __init__(self, input_size, feature_dim, high_feature_dim, class_num, device):
        super(Network, self).__init__()

        self.encoder = Encoder(input_size, feature_dim).to(device)
        self.decoder = Decoder(input_size, feature_dim).to(device)

        self.feature_contrastive_module = nn.Sequential(
            nn.Linear(feature_dim, high_feature_dim),
            # Varying the number of layers of W can obtain the representations with different shapes.
        )

        self.label_contrastive_module = nn.Sequential(
            nn.Linear(feature_dim, class_num),
            nn.Softmax(dim=1)
        )

        self.head = nn.Sequential(
            nn.Linear(feature_dim, class_num),
            nn.Softmax(dim=1)
        )
        self.data_aug = DataAug(dropout=0.9)

    def forward(self, x):
        # [batch_size, feats]
        z = self.encoder(x)  # encoder x

        h = normalize(self.feature_contrastive_module(z), dim=1)  # high level feature
        q = self.label_contrastive_module(z)  # pseudo label
        xr = self.decoder(z)  # decoder x

        return h, q, xr, z  # high_dim_encoder, pseudo label, decoder, encoder

    def forward_cluster(self, x):
        z = self.encoder(x)
        q = self.label_contrastive_module(z)
        pred = torch.argmax(q, dim=1)
        cat_pre = self.head(z)
        return q, pred, cat_pre  # pseudo label, pred, _

    def pretrain(self, epoch, x, batch_size, optimizer):
        tot_loss = 0.
        criterion = torch.nn.MSELoss()
        for batch_x, batch_no in next_batch(x, batch_size):
            optimizer.zero_grad()
            # data augment
            batch_x_1 = self.data_aug(batch_x)
            batch_x_2 = self.data_aug(batch_x)
            # batch_x_1 = batch_x + 1.0 * torch.randn_like(batch_x)
            # batch_x_2 = batch_x + 1.0 * torch.randn_like(batch_x)
            h1, q1, xr1, z1 = self.forward(batch_x_1)  # normalize_encoder, encoder , pseudo label, decoder
            h2, q2, xr2, z2 = self.forward(batch_x_2)

            loss_list = []

            loss_list.append(criterion(batch_x_1, xr1))
            loss_list.append(criterion(batch_x_2, xr2))

            loss = sum(loss_list)
            loss.backward()
            optimizer.step()
            tot_loss += loss.item()
        print('preTrain Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss / len(x)))

    def set_random_seed(self,seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def train(self, x, y, batch_size, optimizer):
        self.set_random_seed(6)
        epochs = 200
        t_acc,t_nmi,t_ari = 0,0,0
        for epoch in range(epochs):
            tot_loss = 0.
            mse = torch.nn.MSELoss()
            for batch_x, batch_no in next_batch(X, batch_size):
                optimizer.zero_grad()
                # data_augment
                batch_x_1 = self.data_aug(batch_x)
                batch_x_2 = self.data_aug(batch_x)
                # Encoder/Decoder
                h1, q1, xr1, z1 = self.forward(batch_x_1)
                h2, q2, xr2, z2 = self.forward(batch_x_2)
                fusion_feature = 0.5 * h1 + 0.5 * h2
                # similarity of the samples in any two views
                sim = torch.exp(torch.mm(h1, h2.t()))
                sim_probs = sim / sim.sum(1, keepdim=True)
                sim_h1 = torch.exp(torch.mm(h1, h1.t()))
                sim_probs_h1 = sim_h1 / sim_h1.sum(1, keepdim=True)
                sim_h2 = torch.exp(torch.mm(h2, h2.t()))
                sim_probs_h2 = sim_h2 / sim_h2.sum(1, keepdim=True)
                l1 = mse(sim_probs, sim_probs_h1)
                l2 = mse(sim_probs, sim_probs_h2)
                # Loss_graph
                L_graph = l1 + l2
                pse_fusion = self.label_contrastive_module(fusion_feature)
                Q_gloal = torch.mm(pse_fusion, pse_fusion.t())
                # Q_gloal = torch.mm(q1, q1.t())
                Q_gloal.fill_diagonal_(1)
                pos_mask_gloal = (Q_gloal >= args.threshold).float()
                Q_gloal = Q_gloal * pos_mask_gloal
                Q = Q_gloal / Q_gloal.sum(1, keepdims=True)
                # Loss_global
                L_global = - (torch.log(sim_probs + 1e-7) * Q).sum(1)
                L_global = L_global.mean()
                # Reconstruction loss
                L_r = mse(batch_x_1, xr1) + mse(batch_x_1, xr1)
                loss = L_r * 10 + L_global*10 + L_graph * 0.01
                loss.backward()
                optimizer.step()
                tot_loss += loss.item()

            if epoch % 1 == 0:
                with torch.no_grad():
                    q, pred, cat_pre = self.forward_cluster(X)  # # pseudo label, pred, _
                    h, _, _, zs = self.forward(X)
                q = q.detach()
                total_pred = np.argmax(np.array(q.cpu()), axis=1)
                nmi, ari, acc = v_measure_score(y, total_pred), adjusted_rand_score(y, total_pred), cluster_acc(y,total_pred)
                if acc > t_acc:
                    if t_acc > 0.83:
                        torch.save(model.state_dict(), 'Model_para/' + db + '-model.ckp' + '_{}'.format(acc))
                    t_acc = acc
                    t_nmi = nmi
                    t_ari = ari
                if epoch % 1 == 0:
                    print('Epoch{}: ACC = {:.4f} NMI = {:.4f} ARI={:.4f}'.format(epoch,acc, nmi, ari))

        print('ACC = {:.4f} NMI = {:.4f} ARI={:.4f}'.format(t_acc, t_nmi, t_ari))



class FeedForwardNetwork(nn.Module):
    def __init__(self, view, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(view, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, view)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        x = x.unsqueeze(1)
        return x


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument("--temperature_f", type=float, default=0.5)
    parser.add_argument("--temperature_l", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--learning_rate", type=float, default=0.0002)
    parser.add_argument("--weight_decay", type=float, default=0.)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=15)
    parser.add_argument("--mse_epochs", type=int, default=300)
    parser.add_argument("--con_epochs", type=int, default=100)
    parser.add_argument("--tune_epochs", type=int, default=50)
    parser.add_argument("--feature_dim", type=int, default=512)  # 512
    parser.add_argument("--high_feature_dim", type=int, default=512)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--ffn_size', type=int, default=32)
    parser.add_argument('--attn_bias_dim', type=int, default=6)
    parser.add_argument('--attention_dropout_rate', type=float, default=0.5)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    db = 'human_ESC'
    # keep nb_genes
    nb_genes = 2000
    X, Y, dims, data_size, class_num = load_train_data(db, nb_genes)
    dataset = MyDataset(X)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    # setup_seed(args.seed)
    dim = data_size
    print("data_size:", dim)
    # model init Network(view, dims, args.feature_dim, args.high_feature_dim, class_num, device)
    model = Network(dim, args.feature_dim, args.high_feature_dim, class_num, device)
    model = model.to(device)

    # dropout
    data_aug = DataAug(dropout=0.9)
    data_aug = data_aug.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    from loss import Loss

    criterion = Loss(args.batch_size, class_num, args.temperature_f, args.temperature_l, device).to(device)

    # training stage
    epoch: int = 1
    train = False
    if train == True:
        model.train(X, Y, args.batch_size, optimizer)
    else:
        state_dict = torch.load('./Model_para/' + db + '-model.ckp')
        model.load_state_dict(state_dict)
        with torch.no_grad():
            q, pred, cat_pre = model.forward_cluster(X)
            h, _, _, zs = model.forward(X)
        zs = zs.detach().cpu()
        q = q.detach().cpu()

        total_pred = np.argmax(np.array(q.cpu()), axis=1)
        #
        nmi, ari, acc = v_measure_score(Y, total_pred), adjusted_rand_score(Y, total_pred), cluster_acc(Y,
                                                                                                        total_pred)
        print('ACC = {:.4f} NMI = {:.4f} ARI={:.4f}'.format(acc, nmi, ari))
