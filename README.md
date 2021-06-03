# OC-HDC

This repository is the official implementation of [OC-HDC: One-Class Hyperdimensional Computing for Outlier Detection](https://openreview.net/attachment?id=jUG3DQhUHve&name=pdf). 

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Training & testing

To try the model(s) in the paper, run this command:

```train & test OC-HDC
python OC-HDC.py --data satimage-2 --seed 1 --level 100 --a 1 --b 1 --epochs 10 --lr 5
```
* data: Selected dataset
* seed: Random seed
* level: Quantization level
* epochs: Retraining iterations
* lr: Learning rate

Other baseline detector

```train & test Isolation Forest
python iforest.py --data satimage-2 --seed 1
```

```train & test OC-SVM
python OC-SVM.py --data satimage-2 --seed 1
```

```train & test DNN-AE
python DNN.py
```



## Results

Our model achieves the following performance:

* test on satimage-2 dataset

|       Dataset      |     Accruacy     |     F1-score    |     Average precision     |     ROC-AUC      |
| ------------------ | ---------------- | --------------- | ------------------------- | ---------------- |
|     satimage-2     |      93.3%       |      87.2%      |          78.4%            |      92.7%       |

