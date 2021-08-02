export AIRBERT_ROOT=$(pwd)
export PYTHONPATH=${PYTHONPATH}:${AIRBERT_ROOT}/build

name=R2R-RC-VLN-BERT/train-sample-aug-init.airbert

flag="--vlnbert vilbert 

      --test_only 0
      --train auglistener
      --init_bert_file snap/vln-bert/r2rM_bnbMS_2capt.pth1.4.bin

      --features places365
      --aug data/prevalent/prevalent_aug.json

      --maxAction 15
      --batchSize 8
      --feedback sample

      --lr 1e-5
      --log_every 2000
      --iters 300000
      --optim adamW
      --mlWeight 0.20
      --maxInput 80
      --angleFeatSize 128
      --featdropout 0.4
      --dropout 0.5"

mkdir -p snap/$name
CUDA_VISIBLE_DEVICES=$1 python r2r_src/train.py $flag --name $name
