device=0

python3 2pretrain.py --gpu $device --dataset cora
python3 2pretrain.py --gpu $device --dataset citeseer
python3 2pretrain.py --gpu $device --dataset ogbn-arxiv
python3 2pretrain.py --gpu $device --dataset reddit
