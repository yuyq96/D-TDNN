# Densely Connected Time Delay Neural Network

PyTorch implementation of Densely Connected Time Delay Neural Network (D-TDNN) in our paper ["Densely Connected Time Delay Neural Network for Speaker Verification"](https://www.isca-speech.org/archive/Interspeech_2020/abstracts/1275.html) (INTERSPEECH 2020).

We provide the [pretrained models](https://github.com/yuyq96/D-TDNN/releases/tag/models) which can be used in many tasks such as:

- Speaker Verification
- Speaker Adaption for Speech Recognition
- Speaker-Dependent Speech Separation
- Multi-Speaker Text-to-Speech

![D-TDNN & D-TDNN-SS](figure/D_TDNN.png)

## Usage

#### Data preparation

- Install [Kaldi](https://github.com/kaldi-asr/kaldi) toolkit.
- Download [VoxCeleb1](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html) test set and unzip it.
- Place `prepare_voxceleb1_test.sh` under `$kaldi_root/egs/voxceleb/v2` and change the `$datadir` and `$voxceleb1_root` in it.
- Run `chmod +x prepare_voxceleb1_test.sh && ./prepare_voxceleb1_test.sh` to generate [30-dim MFCCs](https://github.com/kaldi-asr/kaldi/blob/master/egs/voxceleb/v2/conf/mfcc.conf).
- Place the [`trials`](https://github.com/yuyq96/D-TDNN/releases/download/trials/trials) under `$datadir/test_no_sil`.

#### Test

- Download the pretrained [D-TDNN](https://github.com/yuyq96/D-TDNN/releases/download/models/dtdnn.pth) model and run:
```
python evaluate.py --root $datadir/test_no_sil --model D-TDNN --checkpoint dtdnn.pth --device cuda
```

## Evaluation

VoxCeleb1-O

| Model | Emb. | Params (M) | Loss | Backend | EER (%) | DCF_0.01 | DCF_0.001 |
| :---- | :--: | :--------: | :--: | :-----: | :-----: | :------: | :-------: |
| [TDNN](https://github.com/yuyq96/D-TDNN/releases/download/models/tdnn.pth) | 512 | 4.2 | Softmax | PLDA | 2.34 | 0.28 | 0.38 |
| E-TDNN | 512 | 6.1 | Softmax | PLDA | 2.08 | 0.26 | 0.41 |
| F-TDNN | 512 | 12.4 | Softmax | PLDA | 1.89 | 0.21 | 0.29 |
| [D-TDNN](https://github.com/yuyq96/D-TDNN/releases/download/models/dtdnn.pth) | 512 | 2.8 | Softmax | Cosine | 1.81 | 0.20 | 0.28 |
| D-TDNN-SS (0) | 512 | 3.0 | Softmax | Cosine | 1.55 | 0.20 | 0.30 |
| D-TDNN-SS | 512 | 3.5 | Softmax | Cosine | 1.41 | 0.19 | 0.24 |
| D-TDNN-SS | 128 | 3.1 | AAM-Softmax | Cosine | 1.22 | 0.13 | 0.20 |

## Citation

If you find D-TDNN helps your research, please cite
```
@inproceedings{DBLP:conf/interspeech/YuL20,
  author    = {Ya-Qi Yu and
               Wu-Jun Li},
  title     = {Densely Connected Time Delay Neural Network for Speaker Verification},
  booktitle = {Annual Conference of the International Speech Communication Association (INTERSPEECH)},
  year      = {2020}
}
```
