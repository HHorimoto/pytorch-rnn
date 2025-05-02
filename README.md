# Recurrent Neural Network (RNN) with Pytorch

## Usage

```bash
$ git clone git@github.com:HHorimoto/pytorch-rnn.git
$ cd pytorch-rnn
$ wget https://drive.google.com/uc?id=1oMM1Xu2-hIe4Of2mfznvBNGCQIe54O1f -O ./data/BEMS_data.zip
$ unzip -q -o ./data/BEMS_data.zip -d ./data/
$ ~/python3.10/bin/python3 -m venv .venv
$ . .venv/bin/activate
$ pip install -r requirements.txt
$ source run.sh
```

## Features

### Recurrent Neural Network (RNN)
I trained a model with **RNN**, **GRU**, and **LSTM** using Building Energy Management System (BEMS) dataset.
The table and figure below present the experimental results.

|      |   MSE    | Time (s) |
| ---- | :------: | -------- |
| RNN  | 0.003050 | 325.3228 |
| GRU  | 0.001944 | 330.9667 |
| LSTM | 0.002384 | 324.5983 |

#### Reference
[1] [https://github.com/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/master/13_rnn/01_03_RNN.ipynb](https://github.com/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/master/13_rnn/01_03_RNN.ipynb)
