export AIRBERT_ROOT=$(pwd)
export PYTHONPATH=${PYTHONPATH}:${AIRBERT_ROOT}/build

name=REVERIE-RC-VLN-BERT/train-init.airbert

flag="--vlnbert vilbert

      --train listener
      --test_only 0

      --init_bert_file snap/vln-bert/r2rM_bnbMS_2capt.pth1.4.bin

      --features places365
      --maxAction 15
      --maxInput 50
      --batchSize 4
      --feedback sample
      --lr 1e-5
      --iters 200000
      --log_every 1000
      --optim adamW

      --mlWeight 0.20
      --angleFeatSize 128
      --featdropout 0.4
      --dropout 0.5"

mkdir -p snap/$name
CUDA_VISIBLE_DEVICES=$1 python reverie_src/train.py $flag --name $name
