# Open-world-Deepfake-Detection-Networ(ODDN)
<div align=center>
<img width="350" alt="1723450312316" src="https://github.com/user-attachments/assets/44461f22-304a-45d1-804b-197a6c2fa154">  
</div>

>  ❗️❗️❗️ **News:**
> 
> ✨:1. **Accepted by AAAI 2025 (Oral)**: Our research paper has been accepted by AAAI2025 and chosen as oral presentaion. And latest paper is released at https://arxiv.org/abs/2410.18687. We presents a novel approach designed to address the challenges of deepfake detection in open-world scenarios, particularly on online social networks where unpaired data is prevalent.


## ⏳ Quick Start

### 1. Installation
You can run the following script to configure the necessary environment:

```
git clone https://github.com/ManyiLe/Open-world-Deepfake-Detection-Network.git
cd Open-world-Deepfake-Detection-Network
conda create -n ODDN python=3.9
conda activate ODDN
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install -r requirements.txt
```

### 2. Download Data

<a href="#top">[Back to top]</a>

⭐️ **Datasets** (17 widely used datasets):InfoGAN、BEGAN、CramGAN、AttGAN、MMDGAN、RelGAN、S3GAN、SNGGAN、STGGAN、ProGAN、StyleGAN、StyleGAN2、BigGAN、CycleGAN、StarGAN、GuaGAN、Deepfake

Detailed information about the datasets used in DeepfakeBench is summarized below:

| Dataset | Function | Original Repository |
| --- | --- | --- |
| ForenSynths(ProGAN) | Train | [Hyper-link](https://github.com/PeterWang512/CNNDetection) |
| 8GANs | Test | [Hyper-link](https://github.com/PeterWang512/CNNDetection) |
| 9GANs | Test | [Hyper-link](https://github.com/chuangchuangtan/GANGen-Detection) |


Upon downloading the datasets, please ensure to store them in the [`./datasets`](./datasets/) folder, arranging them in accordance with the directory structure outlined below:

```
datasets
├── train
|   ├── NoCcmp
|   |   ├── airplane
|   |   ├── bicycel
|   |   ├── ......
|   ├── 20%StaticCmp
|   ├── ......
|   ├── other Ratio
├── test
|   ├── AttGAN
|   |   ├── NoCmp
|   │   │   ├──0_real
|   │   │   ├──1_fake
|   |   ├── RandomCmp
|   |   ├── StactioCmp
|   ├── BEGAN
|   ├── ......
├── val
|   ├── Nocmp
|   |   ├── airplane
|   |   ├── bicycle
|   |   ├── ......
|   ├── RandomCmp
|   ├── StaticCmp
```

### 3. Preprocessing
You can run the following script to preprocess images as our experimental setting:

```
python preprocess/random_compression.py -r 0.2 -d 9Gans -m RandomCmp -up 100 -down 30 -m test
python preprocess/random_compression.py -r 0.2 -d 8Gans -m RandomCmp -up 100 -down 30 -m test
python preprocess/random_compression.py -r 0.2 -d 9Gans -m StaticCmp -up 50 -down 50 -m test
python preprocess/random_compression.py -r 0.2 -d 8Gans -m StaticCmp -up 50 -down 50 -m test
python preprocess/random_compression.py -r 0.2 -d ProGan -m RandomCmp -up 100 -down 30 -m train
python preprocess/random_compression.py -r 0.2 -d ProGan -m StaticCmp -up 50 -down 50 -m test
python preprocess/random_compression.py -r 0.2 -d ProGan -m RandomCmp -up 100 -down 30 -m train
python preprocess/random_compression.py -r 0.2 -d ProGan -m StaticCmp -up 50 -down 50 -m test
```
And if you want to try other comfigurations, please adjust the arguments.
### 4. Training

<a href="#top">[Back to top]</a>

```
python train.py
```

You can also adjust the training and testing argument by modifying the config file. By default, the checkpoints and features will be saved during the training process. 

### 5. Evaluation
If you only want to evaluate the detector to produce the results of the cross-dataset evaluation. Here is an example:

```
python eval_test_mygen9GANs.py
python eval_test8gan.py
```

## 📝 Citation

<a href="#top">[Back to top]</a>

If you find our work useful to your research, please cite it as follows:

```
@article{tao2024oddn,
  title={ODDN: Addressing Unpaired Data Challenges in Open-World Deepfake Detection on Online Social Networks},
  author={Tao, Renshuai and Le, Manyi and Tan, Chuangchuang and Liu, Huan and Qin, Haotong and Zhao, Yao},
  journal={arXiv preprint arXiv:2410.18687},
  year={2024}
}

```
