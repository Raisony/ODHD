# OC-HDC

This repository is the official implementation of [OC-HDC: One-Class Hyperdimensional Computing for Outlier Detection](https://openreview.net/attachment?id=jUG3DQhUHve&name=pdf). 

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

>📋  Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc...

## Training & testing

To try the model(s) in the paper, run this command:

```train
python OC-HDC.py --data satimage-2 --seed 1 --level 100 --a 1 --b 1 --epochs 10 --lr 5
```
* data: selected dataset
* seed: random seed
* level: quantization level
* epochs: retraining iterations
* lr: learning rate
* 
## Results

Our model achieves the following performance:

* test on satimage-2 dataset

|       Dataset      |     Accruacy     |     F1-score    |     Average precision     |     ROC-AUC      |
| ------------------ | ---------------- | --------------- | ------------------------- | ---------------- |
|     satimage-2     |      93.3%       |      87.2%      |          78.4%            |      92.7%       |


## Contributing


>📋  Pick a licence and describe how to contribute to your code repository. 
