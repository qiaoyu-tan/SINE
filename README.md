# Sparse-Interest Network for Seqeuntial Recommendation

Original implementation for paper [Sparse-Interest Network for Sequential Recommendation](https://arxiv.org/pdf/2102.09267.pdf).

Qiaoyu Tan, Jianwei Zhang, Jiangchao Yao, Ninghao Liu, Jingren Zhou, Hongxia Yang, Xia Hu

Accepted to WSDM 2021

## Prerequisites

- Python 3.6
- TensorFlow-GPU == 1.15.0rc1
- Faiss-GPU == 1.6.4

## Getting Started

### Installation

- Install TensorFlow-GPU 1.15.0rc1

- Install Faiss-GPU based on the instructions here: https://github.com/facebookresearch/faiss/blob/master/INSTALL.md

- Clone this repo `git clone https://github.com/Qiaoyut/SINE.git`.

### Dataset

- Original links of datasets are:
    
  - https://grouplens.org/datasets/movielens/
  - http://jmcauley.ucsd.edu/data/amazon/index.html
  - https://tianchi.aliyun.com/dataset/dataDetail?dataId=649

- Two preprocessed datasets (MovieLens and Taobao) are included. 

### Training

#### Training on the existing datasets

You can use `python main.py --dataset {dataset_name}` to train SINE on a dataset. Other hyperparameters can be found in the code. 

For example, you can use `python main.py --dataset ml1m` to train SINE model on movieLens dataset.


#### Training on your own datasets

If you want to train models on your own dataset, you should prepare the following three files:
- train/valid/test file: Each line represents an interaction, which contains three numbers `<user_id>,<item_id>,<time_stamp>`.

## Acknowledgement

The structure of our code is based on [MIMN](https://github.com/UIC-Paper/MIMN).

## Cite

Please cite our paper if you find this code useful for your research:

```
@inproceedings{tan2021sparse,
  title={Sparse-interest network for sequential recommendation},
  author={Tan, Qiaoyu and Zhang, Jianwei and Yao, Jiangchao and Liu, Ninghao and Zhou, Jingren and Yang, Hongxia and Hu, Xia},
  booktitle={Proceedings of the 14th ACM International Conference on Web Search and Data Mining},
  pages={598--606},
  year={2021}
}
```
