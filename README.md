# Bird_cocktail
_Stanford CS230 Win2018 project_

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

---
## Introduction
Author, contact, motivation, goal, etc. (citation?)
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
2. Macaulay Library of The Cornell Lab of Ornithology (hereafter ML): [https://www.macaulaylibrary.org](https://www.macaulaylibrary.org)

#### Download and Conversion
This [notebook](https://github.com/Mipanox/Bird_cocktail/blob/master/notebooks/data_preparation.ipynb) 
summarizes the processes of obtaining the data (`mp3` files) and converting them to `wav` files

_(Note: The conversion step is no longer needed since we now use `librosa` package in processing audios (see [below](https://github.com/Mipanox/Bird_cocktail#data-pre-processing)))_

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
