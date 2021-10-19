# Finetuning Airbert on Downstream VLN Tasks

This repository stores the codebase for finetuning [Airbert](https://github.com/airbert-vln/airbert) on downstream VLN tasks including R2R and REVERIE. The code is based on [Recurrent-VLN-BERT](https://github.com/YicongHong/Recurrent-VLN-BERT). We acknowledge [Yicong Hong](https://github.com/YicongHong) for releasing the Recurrent-VLN-BERT code.

## Prerequisites

1. Follow instructions in [Recurrent-VLN-BERT](https://github.com/YicongHong/Recurrent-VLN-BERT#prerequisites) to setup the environment and download data.

For REVERIE task, we use the same object features ([REVERIE_obj_feats.pkl](https://www.dropbox.com/s/prv9anpjhtcrzsm/REVERIE_obj_feats.pkl?dl=0)) as Recurrent-VLN-BERT for fair comparison. 

2. Download the [trained models](https://drive.google.com/drive/folders/14WKuF80E9tvHJMymNxDbbGdtFbezCmR3?usp=sharing).

## REVERIE 

### Inference
To replicate the performance reported in our paper, load the trained models and run validation:
```bash
bash scripts/valid_reverie_agent.sh 0
```

### Training
To train the model, simply run:
```bash
bash scripts/train_reverie_agent.sh 0
```

## R2R
### Inference
To replicate the performance reported in our paper, load the trained models and run validation:
```bash
bash scripts/valid_r2r_agent.sh 0
```

### Training
To train the model, simply run:
```bash
bash scripts/train_r2r_agent.sh 0
```

## Citation
Please cite our paper if you find this repository useful:
```
@misc{guhur2021airbert,
  title ={{Airbert: In-domain Pretraining for Vision-and-Language Navigation}},
  author={Pierre-Louis Guhur and Makarand Tapaswi and Shizhe Chen and Ivan Laptev and Cordelia Schmid},
  year={2021},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
}
```
