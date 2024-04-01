# SEAL for link prediction

SEAL, a link prediction framework based on [GNN](https://github.com/XuSShuai/GNN_tensorflow).

## 1 - About

This repository is a reference implementation of SEAL proposed in the paper: 

>M. Zhang and Y. Chen, Link Prediction Based on Graph Neural Networks, 
Advances in Neural Information Processing Systems (NIPS-18). [Preprint](https://arxiv.org/pdf/1802.09691.pdf)

SEAL, a novel link prediction framework, to simultaneously learn from local enclosing subgraphs, embedding and attributes. 
Experimentally showed the SEAL achieved unprecedentedly strong performance by comparing to various heuristics, latent feature methods, 
and network embedding algorithms.

## 2 - Version

 - python 3.5.5</br>
 - **networkx 2.0**</br>
 - tensorflow 1.7.0</br>
 - numpy 1.14.2</br>

Python 3.5.6 :: Anaconda, Inc.   
 
 | Package             | Version      |
|---------------------|--------------|
| absl-py             | 0.15.0       |
| astor               | 0.7.1        |
| bleach              | 1.5.0        |
| certifi             | 2020.6.20    |
| chardet             | 4.0.0        |
| colorama            | 0.4.5        |
| cycler              | 0.10.0       |
| Cython              | 0.29.14      |
| decorator           | 5.1.1        |
| gast                | 0.5.3        |
| gensim              | 3.8.3        |
| grpcio              | 1.12.1       |
| html5lib            | 0.9999999   |
| idna                | 2.10         |
| importlib-resources | 3.2.1        |
| joblib              | 0.14.1       |
| kiwisolver          | 1.1.0        |
| Markdown            | 2.6.11       |
| matplotlib          | 3.0.3        |
| mkl-fft             | 1.0.6        |
| mkl-random          | 1.0.1        |
| networkx            | 2.0          |
| node2vec            | 0.4.3        |
| numpy               | 1.15.2       |
| pandas              | 0.25.3       |
| pip                 | 10.0.1       |
| protobuf            | 3.6.0        |
| pyparsing           | 2.4.7        |
| python-dateutil     | 2.9.0.post0 |
| pytz                | 2024.1       |
| requests            | 2.25.1       |
| scikit-learn        | 0.22.2.post1|
| scipy               | 1.4.1        |
| setuptools          | 40.2.0       |
| six                 | 1.16.0       |
| smart-open          | 3.0.0        |
| TBB                 | 0.1          |
| tensorboard         | 1.7.0        |
| tensorflow          | 1.7.1        |
| termcolor           | 1.1.0        |
| tqdm                | 4.64.1       |
| urllib3             | 1.26.9       |
| webencodings        | 0.5.1        |
| Werkzeug            | 1.0.1        |
| wheel               | 0.37.1       |
| wincertstore        | 0.2          |
| zipp                | 1.2.0        |


## 3 - Basic Usage

#### 3.1 - Example

Type the following command to run `seal` on data 'USAir'.

```python
python main.py
```

#### 3.2 - Option

 - `python main.py --data Celegans` to run `SEAL` on data `Celegans`
 - `python main.py --epoch 200` will assign the number of epochs to 200, default value is 100
 - `python main.py -r 0.00001` will set the learning rate which determine the speed of update parameters to 0.00001.

you can check out the other options available to use `python main.py --help`

## 4 - Result

|Data| USAir | Celegans | Power | Yeast | PB | Router |
|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
|#Node         |332    |297    |4941| 2375 | 1222 | 5022 |
|#Edges        |2126   |2148   |6594| 11693 | 16714 | 6258 |
|Average Degree|12.8072|14.4646|2.6691|9.8467| 27.3553| 2.4922 |
|SEAL(**auc**)     |**0.9538**|**0.8979**|**0.8889**|**0.9714**|**0.9444**|**0.9412**|
