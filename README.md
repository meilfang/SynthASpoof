# SPL-MAD: Unsupervised Face Morphing Attack Detection via Self-paced Anomaly Detection
---
## Note
This is the official repository of the paper: Unsupervised Face Morphing Attack Detection via Self-paced Anomaly Detection, accepted at IJCB2022.

## Motivation
We studied the behavior of unsupervised anomaly detection through reconstruction error analyses on MAD data. Our study revealed that morphing attacks are more straightforward to reconstruct than bona fide samples when the reconstruction is learned on general face data. Then, we leverage the above-stated observation to present a novel unsupervised MAD solution via an adapted self-paced anomaly detection, namely SPL-MAD. The adapted SPL paradigm proved helpful in neglecting the suspicious unlabeled data in training and thus enhancing the reconstruction gap between bona fide and attack samples, leading to improving the generalizability of the MAD model.
## Data preparation
### Dataset
*CASIA-WebFace:* The CASIA-WebFace dataset is used for face verification and face identification tasks. The dataset contains 494,414 face images of 10,575 real identities collected from the web.
*SMDD*: SMDD is a synthetic face morphing attack detection development dataset. Additional morphing attack data is used as data containimitaion in our case. SMDD can be downloaded via [SMDD Github](https://github.com/naserdamer/SMDD-Synthetic-Face-Morphing-Attack-Detection-Development-dataset).


## Training
Example of training:
```
python train.py \
  --train_csv 'training_data.csv' \
```
where training_data.csv contains image path.
## Testing
Example of testing:
```
python train.py \
  --test_csv 'test_data.csv' \
  --model_path 'casia_smdd.pth'
```
where test_data.csv contains image path and the corresponding label (bonafide or attack).

## Models
The model trained on CAISA-WebFace and the training set of SMDD can be download via [google driver](https://drive.google.com/file/d/1kFLp1dWp_sBwC-l-RTVo-LRitKSYxbyv/view?usp=sharing).
More information and small test can be found in test.py. Please make sure give the correct model path.

if you use SPL-MAD in this repository, please cite the following paper:
```
@inproceedings{splmad,
  author    = {Meiling Fang and
              Fadi Boutros and
              Naser Damer},
  title     = {Unsupervised Face Morphing Attack Detection via Self-paced Anomaly Detection},
  booktitle = {{IJCB}},
  pages     = {1--8},
  publisher = {{IEEE}},
  year      = {2022}
}
```
if you use SMDD dataset, please cite the following paper:
```
@article{SMDD,
    author    = {Damer, Naser and L\'opez, C\'esar Augusto Fontanillo and Fang, Meiling and Spiller, No\'emie and Pham, Minh Vu and Boutros, Fadi},
    title     = {Privacy-Friendly Synthetic Data for the Development of Face Morphing Attack Detectors},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2022},
    pages     = {1606-1617}
}
```


## License
The implementation or trained models use is restricted to research purpuses. The use of the implementation/trained models for product development or product competetions (incl. NIST FRVT MORPH) is not allowed. This project is licensed under the terms of the Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) license. Copyright (c) 2020 Fraunhofer Institute for Computer Graphics Research IGD Darmstadt.
