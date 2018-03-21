# Cocktail Party Problem of Bird Sounds
_Stanford CS230 Win2018 project_

A Convolutional-Neural-Network-based project to tackle multi-class multi-label classification of bird sounds. Implemented in [PyTorch](http://pytorch.org)

_(Dated: 03/20/2018)_

## Content
### Introduction
### 0. Installation and Prerequisites
  - [Installation](https://github.com/Mipanox/Bird_cocktail#installation)
### 1. Data
  - [Data Preparation](https://github.com/Mipanox/Bird_cocktail#data-preparation)
  - [Data Pre-processing](https://github.com/Mipanox/Bird_cocktail#data-pre-processing)
  - [Data Augmentation and Mixing](https://github.com/Mipanox/Bird_cocktail#data-augmentation-and-sampling)
### 2. Train the Model
  - [Build train/val/test sets](https://github.com/Mipanox/Bird_cocktail#build-datasets)
  - [Training](https://github.com/Mipanox/Bird_cocktail#training)
  - [Hyperparameter Tuning](https://github.com/Mipanox/Bird_cocktail#hyperparameter-tuning)
  
### 3. Evaluation
  - [Synthesize Results from experiments](https://github.com/Mipanox/Bird_cocktail#synthesize-results)
  - [Evaluation on Test Set](https://github.com/Mipanox/Bird_cocktail#evaluation-on-test-set)
  
### 4. Performance
  - [Our Current Progress](https://github.com/Mipanox/Bird_cocktail#our-current-progress)
  - [Single-label Benchmark](https://github.com/Mipanox/Bird_cocktail#single-label-benchmark)
  - [Multi-label Model Comparison](https://github.com/Mipanox/Bird_cocktail#multi-label-model-comparison)
  
### 5. Epilogue
  - [Future Perspectives](https://github.com/Mipanox/Bird_cocktail#future-perspectives)
  
### 6. [References](https://github.com/Mipanox/Bird_cocktail#references)

---
## Introduction
### Information
Authors: Jason Chou and Chun-Hao To - Stanford University, CA, USA

E-Mail Contact: jasonhc@stanford.edu

If you use the content of this work, please consider citing this code repository with the following BibTeX or plaintext entry. The BibTeX entry requires the ```url``` LaTeX package.

```
@misc{birdcocktail,
  title = {{Cocktail Party Problem of Bird Sounds}},
  author = {Chou, Jason and To, Chun-Hao},
  howpublished = {\url{https://github.com/Mipanox/Bird_cocktail}},
  note = {Accessed: [Insert date here]}
}

Chou, Jason and To, Chun-Hao
Cocktail Party Problem of Bird Sounds
https://github.com/Mipanox/Bird_cocktail
Accessed: <Insert date here>
```

### Prologue
The field of bird song recognition has recently seen thriving developments thanks to the rapid growth of the database for bird songs and calls. Not only are there regular workshops held internationally, but also downloadable mobile apps [[o1](https://github.com/Mipanox/Bird_cocktail#others)], both showing promising identification capabilities of current implementations. Despite outperforming (non-expert) humans in this task of recognizing bird species by sounds, all existing implementations flounder when it faces noisy background and/or multiple birds in the sound recordings – they fail when more than one species are to be singly identified within one recording.

Therefore, in this project, we endeavor to improve the existing algorithms by adding to them the power of separating distinct species within the cacophonies, an avian analogue of solving the cocktail party problem. Challenging as the problem may be, the success of the project can potentially benefit bird lovers and researchers by becoming a more practical software that does not require recordings to be pre-processed or cleaned – you record something, it outputs the answers right away. The project is thus a multi-class multi-label classification problem of recognizing constituents from mixtures of classes.

One should note that, although we call it cocktail party problem, we do not aim at "extracting" the sound components, but only at being able to identify the individual birds.

---
## 0. Installation and Prerequisites
### Installation
This is a PyTorch implementation of multi-class multi-label classification of hundreds of bird species. We wrote the code in python 2.7, but should be runnable with python 3+. CUDA is optional for the project, but highly recommended to accelerate training process. All required packages can be installed using `pip`, after cloning this repository:

```
git clone https://github.com/Mipanox/Bird_cocktail.git
cd Bird_cocktail
sudo pip install –r requirements.txt
(or pip install --user -r requirements.txt if you are on systems without root access)
```

If you have trouble installing some of the packages, please refer to their official documentation for further instructions.

_[(back to top)](https://github.com/Mipanox/Bird_cocktail#content)_

---
## 1. Data
Before training, obtaining processible data is essential. We choose to play with `mp3` audio files which are converted into spectrograms (images) with proper pre-processing, then are then fed into the network with data augmentation and sampling on the fly.

### Data Preparation
#### Databases
We use the following two databases:
1. xeno-canto (hereafter XC): [https://www.xeno-canto.org](https://www.xeno-canto.org)
2. Macaulay Library of The Cornell Lab of Ornithology (hereafter ML): [https://www.macaulaylibrary.org](https://www.macaulaylibrary.org) _(Note: As of this writing, none of the ML data have been used in either training or evaluation)_

#### Download and Conversion
This [notebook](https://github.com/Mipanox/Bird_cocktail/blob/master/notebooks/data_preparation.ipynb) 
summarizes the processes of obtaining the data (`mp3` files) and converting them to `wav` files

_(Note: The conversion step is no longer necessary since we now use `librosa` package in processing audios (see [below](https://github.com/Mipanox/Bird_cocktail#data-pre-processing)))_

### Data Pre-processing
Because we are tackling the problem of recognizing individual bird species in mixtures of sounds (i.e. multi-label classification), we can synthesize datasets by manually superposing the audio clips of different bird species, with some random weights (relative intensity), etc. We choose to do this on-the-fly in the training process, and in the Fourier domain, viz., from spectrograms. Therefore, we will pre-process our raw data--audios, by transforming them into spectrograms

#### Running the code
This [notebook](https://github.com/Mipanox/Bird_cocktail/blob/master/notebooks/data_preprocessing.ipynb) explains the work behind the code.

After preparing the audio files for individual species as described above (you may otherwise obtain your datasets, but they have to be arranged in the same way), they will have been arranged in this structure:
```
RAW dataset   
¦
+---species1
¦   ¦   spe1_file001.mp3
¦   ¦   spe1_file002.mp3
¦   ¦   ...
¦   
+---species2
¦   ¦   spe2_file001.mp3
¦   ¦   spe2_file002.mp3
¦   ¦   ...
¦    
+---...

```

Then, run the following code, specifying the paths of RAW audios as well as the destination for spectrograms:
1. Serial:
```
python codes/aud_to_spec.py --src_dir <path_to_raw_audios> --spec_dir <path_to_spec_destination>
```

2. Parallel - If run on multiple CPUs, use the other pre-processing code:
```
pytho codes/aud_to_spec_parallel.py --src_dir <path_to_raw_audios> --spec_dir <path_to_spec_destination>
```
_(Note1: We assume that parallel jobs are executed by specifying number of CPUs (and/or nodes), which in principle should be detected automatically by the code. Otherwise, one may need to hardcode the number of CPUs used in the code)_

_(Note2:It is also possible to select different arguments for various functions: e.g. threshold for signal/noise discrimination...)_

### Data Augmentation and Sampling
Although data augmentation and sampling are done on-the-fly when training/evaluating the model, we decide to outline the underlying processes here. Refer to this [notebook](https://github.com/Mipanox/Bird_cocktail/blob/master/notebooks/data_loading.ipynb) for details.


_[(back to top)](https://github.com/Mipanox/Bird_cocktail#content)_

---
## 2. Train the Model
### Build datasets
Up to this point, you should've gone through [obtaining data](https://github.com/Mipanox/Bird_cocktail#data-preparation) (in `mp3` or `wav`) and [pre-processing](https://github.com/Mipanox/Bird_cocktail#data-pre-processing), ending up with a folder containing spectrograms of each species:
```
dataset   
¦
+---species1
¦   ¦   spe1_spec001.jpg
¦   ¦   spe1_spec002.jpg
¦   ¦   ...
¦   
+---species2
¦   ¦   spe2_spec001.jpg
¦   ¦   spe2_spec002.jpg
¦   ¦   ...
¦    
+---...
```
Now, run the next line to shuffle and split the data into train/val/test sets:
```
python build_dataset.py --data_dir <path_to_preprocessed_data> --output_dir <path_to_desired_splitted_datasets>
```
There are two more options that you can tune:
- `--train_per`: Percentage of training set, defaults to 98 (train/val/test = 98/1/1)
- `--max_spec`: Maximum number of species for the model, sorted by richness of data (number of spectrograms available). Defaults to 300. This one should be fixed according to your needs. For example, if you want your model to specialize in the birds in Stanford, you don't need many species (but of course the species in the dataset must come from the same region you're interested in. You're only throwing away "less common" ones or "rarities")

You will then have copied your spectrograms to something like:
```
split_dir   
¦
+---train/
¦   ¦   species1
¦   ¦   ¦   spe1_spec001.jpg
¦   ¦   ¦   spe1_spec002.jpg
¦   ¦   ...
¦   
+---val/
¦   ¦   species1
¦   ¦   ¦   spe1_spec003.jpg
¦   ¦   ¦   spe1_spec004.jpg
¦   ¦   ...
¦    
+---test/

```

### Training

### Hyperparameter Tuning



_[(back to top)](https://github.com/Mipanox/Bird_cocktail#content)_

---
## 6. References
### Papers
[p1] Huang _et al._, Densely Connected Convolutional Networks. (2016). arXiv: [1608.06993](https://arxiv.org/abs/1608.06993)

[p2] He _et al._, Deep Residual Learning for Image Recognition. (2015). arXiv: [1512.03385](https://arxiv.org/abs/1512.03385)

[p3] Li _et al._, Improving Pairwise Ranking for Multi-label Image Classification (2017). arXiv: [1704.03135](https://arxiv.org/abs/1704.03135)

[p4] Szegedy _et al._, Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning. (2016). arXiv: [1602.07261](https://arxiv.org/abs/1602.07261)

### Repositories
[r1] Stanford CS230 Example Code [Repository](https://github.com/cs230-stanford/cs230-code-examples/tree/master/pytorch/vision)

[r2] Kahl _et al._, [Working Notes](http://ceur-ws.org/Vol-1866/paper_143.pdf) of BirdCLEF. (2017). Github [repository](https://github.com/kahst/BirdCLEF2017)

[r3] Amos and Kolter, A PyTorch Implementation of DenseNet. Github [repository](https://github.com/bamos/densenet.pytorch); also andreaazzini's [repository](https://github.com/andreaazzini/multidensenet)

[r4] Tensorflow Model Zoo for Torch7 and PyTorch. Github [repository](https://github.com/Cadene/tensorflow-model-zoo.torch)

### Others
[o1] e.g. [BirdCLEF](http://www.imageclef.org/lifeclef/2017/bird), [EADM](http://sabiod.univ-tln.fr/EADM/), [BirdGenie](http://www.birdgenie.com), [Bird Song Id USA](http://us.isoperlaapps.com/BirdSongIdUSA.html); and proceedings/publications therein

[o2] [xeno-canto](https://www.xeno-canto.org) database of bird sounds

[o3] Official PyTorch [tutorials](http://pytorch.org/tutorials/), [documentation](http://pytorch.org/docs/0.3.1/), and [source code](https://github.com/pytorch/pytorch)

