# SynthASpoof
---
## Note
This is the official repository of the work: .

## SynthASpoof dataset
The SynthASpoof is the synthetic-based face presentation attack detection datasets, including synthetic-generated 25,000 bona fide images and 78,800 corresponding attacks collected by presenting the printed/replayed images to capture cameras (one mobile phone, two different tablets, and one webcam).
The image samples in SynthASpoof are shown:

![grafik](figures/SynPAD_samples.pdf)

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
