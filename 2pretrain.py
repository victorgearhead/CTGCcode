from util.data_loader import *
from util.evaluate import *
from util.hyperpara import *
from util.models import *
from util.module import *
from util.training import *
from util.utils import *
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default="cora", help= 'cora, citeseer, ogbn-arxiv, reddit')
parser.add_argument('--gpu', type=int, default=2)
parser.add_argument('--result_path', type=str, default="./results/")
parser.add_argument('--model_path', type=str, default="./save_pretrain_model/")
parser.add_argument('--eigen_path', type=str, default="./save_eigen/")
parser.add_argument('--data_dir', type=str, default="./data/")
parser.add_argument('--split_data_dir', type=str, default="./dataset_split/")

# pretrain
parser.add_argument('--epoch_pretrain', type=int, default=200)
parser.add_argument('--epoch_ssl', type=int, default=20)
parser.add_argument('--iter_num', type=int, default=5)
parser.add_argument('--lr_pretrain', type=float, default=0.001)
parser.add_argument('--lr_ssl_spa', type=float, default=0.0001)
parser.add_argument('--lr_ssl_spe', type=float, default=0.001)
parser.add_argument('--alpha', type=float, default=1)

# downstream task
parser.add_argument('--reduction_rate', type=float, default=0.005)
parser.add_argument('--epoch_cls', type=int, default=200)
parser.add_argument('--epoch_lp', type=int, default=200)
parser.add_argument('--lr_cls', type=float, default=0.01)
parser.add_argument('--lr_lp', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--nrepeat', type=int, default=5)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--n_dim', type=int, default=256)
parser.add_argument('--eva_iter', type=int, default=1)
parser.add_argument('--test_gnn', type=str, default='GCN')
parser.add_argument('--shot', type=int, default=3)
args = parser.parse_args()
args = device_setting(args)
seed_everything(args.seed)


args.result_path =  f'./results_proposed/'
args.eigen_path +=  f'{args.dataset_name}/'
if not os.path.exists(args.result_path):
    os.makedirs(args.result_path)
if not os.path.exists(args.model_path):
    os.makedirs(args.model_path)

args = SSL_hyperpara(args)
args = SSL_reduction(args)

acc_shot3_NC= []
acc_shot5_NC= []
auc_LP= []
acc_LP= []
nmi_CL = []
ari_CL = []


for i in range(args.nrepeat):
    args.seed += 1

    ## data
    datasets = get_dataset(args)
    args, data, data_val, data_test = set_dataset(args, datasets)
    print("train num:", int(data.train_mask.sum()))
    args.syn_num = int(data.train_num_original * args.reduction_rate)

    ## model
    model_spa = GCN(data.num_features, args.n_dim, args.num_class, 2, args.dropout).to(args.device)

    ## initialization
    model_spa, cluster_idx = pre_train(args, data, data_test, model_spa)

    ## evaluate
    H, H_val, H_test, H_test_masked, labels_test, \
    H_train_shot_3, label_train_shot_3, \
    H_train_shot_5, label_train_shot_5 = eva_data(args, data, data_val, data_test, model_spa)

    if args.dataset_name != "ppi":
        nmi, ari = evaluate_CL(H_test_masked, labels_test)
        nmi_CL.append(nmi)
        ari_CL.append(ari)

    if args.dataset_name != "ppi":
        acc_shot3, acc_shot5 = evaluate_NC(args, H_train_shot_3, label_train_shot_3, H_train_shot_5, label_train_shot_5, H_test_masked, labels_test)
        acc_shot3_NC.append(acc_shot3)
        acc_shot5_NC.append(acc_shot5)

    auc_lp, acc_lp = evaluate_LP(args, data, H, H_val, H_test, data_val, data_test)
    auc_LP.append(auc_lp)
    acc_LP.append(acc_lp) 
    print()

pretrain_record_caption(args)

if args.dataset_name != "ppi":
    result_record_whole_NC(args, acc_shot3_NC, shot=3)
    result_record_whole_NC(args, acc_shot5_NC, shot=5)
    result_record_whole_CL(args, nmi_CL, ari_CL)
else:
    print("\n[INFO] Skipping NC and CL result saving for PPI (multi-label)\n")

result_record_whole_LP(args, auc_LP, acc_LP)
print()