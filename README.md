# MIP: Modality- and Instance-aware Visual Prompts Network
ICMR 2024 Oral | Enhancing Visible-Infrared Person Re-identification with Modality- and Instance-aware Visual Prompt Learning

## Project home
[MIP: Enhancing Visible-Infrared Person Re-identification with Modality- and Instance-aware Visual Prompt Learning](https://wurqjackey.github.io/ICMR2024_MIP/)


## Install
This code is based on the [TransReID](https://github.com/damo-cv/TransReID) project. Please Refer to [README_TransReID.md](https://wurqjackey.github.io/MIP/README_TransReID.md).

## Usage
### Train
```
bash train.sh
```

### Evaluation
```
bash eval_sysu.sh
bash eval_regdb.sh
```
### Checkpoint
You can download our models trained on SYSU-MM01 and RegDB. <br>
BaiduNetdisk: [[Checkpoints]](https://pan.baidu.com/s/1XOzA05ADSfiTaeHAj4sDsA) Code: m312 <br>
GoogleDrive: [[Checkpoints]](https://drive.google.com/drive/folders/1MUawaVku45vviDrxh9rJtshDEL2oCi6C?usp=sharing)

## Acknowledgement
So much thanks for codebase from [TransReID](https://github.com/damo-cv/TransReID).

## Reference
If you find this code useful for your research, please cite our paper.
```
@inproceedings{10.1145/3652583.3658109,
author = {Wu, Ruiqi and Jiao, Bingliang and Wang, Wenxuan and Liu, Meng and Wang, Peng},
title = {Enhancing Visible-Infrared Person Re-identification with Modality- and Instance-aware Visual Prompt Learning},
year = {2024},
isbn = {9798400706196},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3652583.3658109},
doi = {10.1145/3652583.3658109},
booktitle = {Proceedings of the 2024 International Conference on Multimedia Retrieval},
pages = {579–588},
numpages = {10},
keywords = {cross-modality person re-identification, visible-infrared person re-identification, visual prompt learning},
location = {Phuket, Thailand},
series = {ICMR '24}
}
```
