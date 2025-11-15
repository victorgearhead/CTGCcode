echo ""
echo "================================"
echo "  Started (Linux)"
echo "  If no GPU is available, set --gpu -1 in the commands above"
echo "================================"
echo ""


python3 1preprocess.py --gpu 0 --dataset cora
python3 2pretrain.py --gpu 0 --dataset cora

for shot in 3 5
do
    for r in 0.25 0.5 1
    do
        python3 3relaymodel.py --gpu 0 --dataset cora --reduction_rate $r --shot $shot
    done
done

for shot in 3 5
do
    for r in 0.25 0.5 1
    do
        python3 4condense.py --gpu 0 --dataset cora --reduction_rate $r --shot $shot
    done
done


for shot in 3 5
do
    python3 whole.py --gpu 0 --dataset cora --shot $shot
done
python3 whole_LP.py --gpu 0 --dataset cora

echo ""
echo "================================"
echo "  All tasks finished (Linux)"
echo "================================"
echo ""
