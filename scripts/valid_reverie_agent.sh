export AIRBERT_ROOT=$(pwd)
export PYTHONPATH=${PYTHONPATH}:${AIRBERT_ROOT}/build

name=REVERIE-RC-VLN-BERT/init.airbert

flag="--vlnbert vilbert

      --test_only 0

      --train validlistener
      --submit

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

CUDA_VISIBLE_DEVICES=$1 python reverie_src/train.py $flag --name $name

