device=-1

python3 1preprocess.py --gpu $device --dataset cora 
python3 1preprocess.py --gpu $device --dataset citeseer 
python3 1preprocess.py --gpu $device --dataset ogbn-arxiv 
python3 1preprocess.py --gpu $device --dataset reddit 
