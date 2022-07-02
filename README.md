# Cocktail Party Problem of Bird Sounds
_Stanford CS230 Win2018 project_

A Convolutional-Neural-Network-based project to tackle multi-class multi-label classification of bird sounds. Implemented in [PyTorch](http://pytorch.org)

_(Dated: 03/21/2018)_

---
## Summary
More often than not birds move in groups and make sounds together. Being able to recognize individual bird sounds in the natural mixture is thus of great practical values for bird lovers and researchers. We train Convolutional Neural Networks with different schemes on spectrograms of multi-species bird sounds to tackle this multi-class multi-label classiﬁcation problem. Variations of ResNet combined with binary relevance method prove to be the most powerful, while changing loss functions and/or details of network architectures does not help improve the performance signiﬁcantly. For a total of 10 species with 1–5 species present simultaneously in one segment of sound, our models achieve F1 score, precision and recall of 0.83, 0.90 and 0.85 respectively, considerably outperforming similar multi-label classiﬁcation tasks on photo-images.

## Data preparation
### Pre-processing
We obtain `mp3` recordings of 10 common local "loud" bird species from the `xeno-canto` database, each capturing bird songs, calls or other representative sounds (e.g. woodpecker’s drumming). We use magnitude spectrograms as inputs for models. This is done by ﬁrst segmenting recordings into 3-sec chunks, then applying Short-Time Fourier Transform (SFTF) to convert them into spectrograms – a time versus frequency plot, sampled on a grey-scale 2D image of dimensions 128×192. 

We follow the heuristics in Kahl et al.’s work on single-label classiﬁcation of bird sounds for separating signals and noises. The algorithm does median blur, median threshold, spot removal and morphological closing, based on which it judges whether there is strong-enough signal contained in one spectrogram. Figure below illustrates the idea.
<p align="center"><img src="https://github.com/Mipanox/Bird_cocktail/blob/master/images/eg_signal_preproc.png" width="600"/></p> 

### Synthesization
we synthesize our own multi-species spectrograms from the un-processed signal and noise spectrograms. 1 – 5 spectrograms are randomly chosen and weighted, then superposed together with randomly chosen noise sample, to form a single input for model. Mathematics ensures almost countless possible combinations, which means we can have arbitrary augmentation. Finally, a random cropping is optionally done to further augment the data. An example 3-species synthesized spectrogram is shown below:
<p align="center"><img src="https://github.com/Mipanox/Bird_cocktail/blob/master/images/eg_syn_spec_3birds.png" width="500"/></p> 


## Models
Our representative models are combinations of (simpliﬁed) ResNet, DenseNet, and BLSTM, taking advantage of each architecture -- e.g. the sequential nature of sound is exploited in BLSTM.
<p align="center"><img src="https://github.com/Mipanox/Bird_cocktail/blob/master/images/models.png" width="700"/></p> 

## Results
Summary of F1 score, precision, and recall of each neural network. The best values among all neural networks are highlighted in red. For more details, see below.
<p align="center"><img src="https://github.com/Mipanox/Bird_cocktail/blob/master/images/results.png" width="700"/></p> 

---
# Content
## Prologue
### Introduction
  - [Authors, Contact, and Citation](https://github.com/Mipanox/Bird_cocktail#information)
  - [Motivation and Goals](https://github.com/Mipanox/Bird_cocktail#motivation-and-goals)
  
### 0. Installation and Prerequisites
  - [Installation](https://github.com/Mipanox/Bird_cocktail#installation)

## How to Use
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
  
## What to Expect
### 4. Performance
  - [Our Current Progress](https://github.com/Mipanox/Bird_cocktail#our-current-progress)
  - [Single-label Benchmark](https://github.com/Mipanox/Bird_cocktail#single-label-benchmark)
  - [Multi-label Model Comparison](https://github.com/Mipanox/Bird_cocktail#multi-label-model-comparison)
  
### 5. Reproducing the Results
  - [Download Our Datasets](https://github.com/Mipanox/Bird_cocktail#download-our-datasets)
  - [Using the Pre-trained Models](https://github.com/Mipanox/Bird_cocktail#using-the-pre-trained-models)

### 6. Epilogue
  - [Future Perspectives](https://github.com/Mipanox/Bird_cocktail#future-perspectives)
  
### 7. [References](https://github.com/Mipanox/Bird_cocktail#7-references-1)

---
# Prologue
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

### Motivation and Goals
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
# How to Use
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
To train a model, first create a directory for under the experiments directory (or anywhere else you like). It should contain a file ```params.json``` which sets the hyperparameters for the experiment. Some examples can be found in the [directory](https://github.com/Mipanox/Bird_cocktail/tree/master/experiments).

Run the following line to start the experiment, specifying the path to the hyperparameter ```params.json``` file:
```
python train.py --data_dir <path_to_splitted_datasets> --model_dir <path_to_the_folder_of_json_file>
```
It will instantiate a model and train it on the training set following the hyperparameters specified in ```params.json```. It will also evaluate some metrics on the validation set. The results will be stored in the same folder containing `params.json`, including metrics and validation sets, training logs, tensorboard logs. You can take a look inside the `train.py` as well as the `evaluate.py` codes for more details.

Currently, we support training with the following models, loss functions, optimizers, and modes _(with the relevant hyperparameters given)_:

- **Modes**
  1. Single-label: `if_single: 1`. Only uses cross-entropy loss. Binary relevance models listed below are not suitable
  2. Multi-label: `(none; default)`
- **Models** _(each with their own tunable hyperparameters)_
  1. DenseNet [[p1,r3,o3]](https://github.com/Mipanox/Bird_cocktail#papers): `model: 1`
  2. SqueezeNet [[p5,o3]](https://github.com/Mipanox/Bird_cocktail#papers): `model: 2`
  3. Inception-v4 [[p5,r4]](https://github.com/Mipanox/Bird_cocktail#papers): `model: 3`
  4. InceptionResNet [[p5,r4]](https://github.com/Mipanox/Bird_cocktail#papers): `model: 4`
  5. ResNet [[p2,o3]](https://github.com/Mipanox/Bird_cocktail#papers): `model: 5`
  6. DenseNet+BinaryRelevance (DenseNet+BR): `model: 6`
  7. ResNet+BinaryRelevance (ResNet+BR): `model: 7`
  8. DensNet+BLSTM: `model: 8`
- **Loss Functions**
  1. Binary-Cross-Entropy (BCE) loss: `(none; default)`
  2. Weighted-Approximate-Ranking Pairwise (WARP) loss [[p7]](https://github.com/Mipanox/Bird_cocktail#papers): `loss_fn: 1`
  3. Log-Sum-Exponential Pairwise (LSEP) loss [[p3]](https://github.com/Mipanox/Bird_cocktail#papers): `loss_fn: 2`
- **Optimizers**
  1. Adam: `optimizer: 1` with optional L2 penalty `alpha`
  2. Stochastic Gradient Descent (SGD): `optimizer: 2` with optional L2 penalty `alpha`

**Note:** For single/multi-label training, we use accuracy/F1 score as the reference metrics in determining "best" performance. Namely, a model is considered "improved" when the accuracy/F1 score has surpassed previous highest value.

### Hyperparameter Tuning
Fine tuning a model is as simple as running another experiment. Simply create another directory containing another `params.json`, and run the training script above with the proper path.

_[(back to top)](https://github.com/Mipanox/Bird_cocktail#content)_

---
## 3. Evaluation
### Synthesize Results
After exhausting the computing resources with extensive hyperparemeter searches, you should have had a bunch of different models with varying capabilities now. You may want to juxtapose all of them in a table, for the performance on validation set. Call this:
```
python synthesize_results.py --parent_dir <path-to-parent-folder-with-various-exps>
```

The code will generate a table which summarizes the metrics on the validation set. Something like:

|               model                                    |       f1 |   recall |        loss |   precision |   accuracy |
|:-------------------------------------------------------|---------:|---------:|------------:|------------:|-----------:|
| experiments/multi-label/densenet_03    | 0.579357 | 0.712428 | 0.453257    |    0.528843 |   0.773568 |
| experiments/multi-label/densenet_02    | 0.503218 | 0.669583 | 0.529402    |    0.442973 |   0.718413 |
| ... | ... | ... | ... | ... | ...|

### Evaluation on Test Set
Once you've run many experiments and selected your best model and hyperparameters based on the performance on the validation set, you can finally evaluate the performance of your model on the test set:
```
python evaluate.py --data_dir <path-to-test-data> --model_dir <path-to-folder-of-the-selected-model>
```

For single-label problem specifically, we have confusion matrix computed for you. Upon completing either training or evaluation, confusion matrices will be stored in folders like `cm_val` or `cm_test`. You can plot each of them using the function in `utils.py` by calling
```
import utils
cm = np.load('<path-to-npy-file')
utils.plot_cm(cm)
```
with other optional plotting options.

_[(back to top)](https://github.com/Mipanox/Bird_cocktail#content)_

---
## 4. Performance
### Our Current Progress
We were developing, training, and testing on AWS's Deep Learning AMI for 10 common "loud" species in the Bay Area (CA, USA). The 10 species are: Acorn Woodpecker, American Crow, American Goldfinch, American Robin, Bewick's Wren, Fox Sparrow, Hermit Thrush, Song Sparrow, Spotted Tohwee, and White-throated Sparrow. In the [section](https://github.com/Mipanox/Bird_cocktail#download-our-datasets) below, we also provide download link to the pre-processed data we used.

These 40,000 spectrograms correspond to roughly 25+ hours of field recordings. They were all obtained and processed with the pipeline outlined above.

### Single-label Benchmark
Quoting the best-ever-achieved results, the [ResNet_02](https://github.com/Mipanox/Bird_cocktail/tree/master/experiments/AWS-experiments/single-label/resnet_02_new) achieved 1.000 and 0.942 accuracy for training and validation sets respectively. Below shows an example training curve (in the middle of training):
<p align="center"><img src="https://github.com/Mipanox/Bird_cocktail/blob/master/images/sing_res_07_tb.png" width="500"/></p> 

and the best confusion matrix evaluated on test set (notice the log-scale colorbar):

<p align="center"><img align="center" src="https://github.com/Mipanox/Bird_cocktail/blob/master/images/sing_res_02_cm.png" width="450"/></p> 


### Multi-label Model Comparison
The following table summarizes the performance of some noteworthy models on validation set (not meant to be complete!):


|               model        |   loss |       f1 |   recall |   precision | regularization | comments |
|:--------------------------:|:---------:|:--------:|:--------:|:-----------:|:--------------:|:---------|
| [ResNet+BR](https://github.com/Mipanox/Bird_cocktail/tree/master/experiments/AWS-experiments/multi-label/resnet_br_02_3_) | BCE | **0.831** | 0.804 | **0.904** | None | multiple stages of training from [this](https://github.com/Mipanox/Bird_cocktail/tree/master/experiments/AWS-experiments/multi-label/resnet_br_02) and [this](https://github.com/Mipanox/Bird_cocktail/tree/master/experiments/AWS-experiments/multi-label/resnet_br_02_2) |
| [DenseNet](https://github.com/Mipanox/Bird_cocktail/tree/master/experiments/AWS-experiments/multi-label/lsep_dense_06) | LSEP | 0.735 | 0.686 | 0.856 | Early Stopping | high variance |
| [DenseNet+LSTM](https://github.com/Mipanox/Bird_cocktail/tree/master/experiments/AWS-experiments/multi-label/densenet_blstm) | LSEP | 0.654 | 0.579 | 0.651 | L2 | |
| [ResNet](https://github.com/Mipanox/Bird_cocktail/tree/master/experiments/AWS-experiments/multi-label/lsep_resnet_01) | LSEP | 0.793 | 0.759 | 0.830 | L2 | |
| [ResNet](https://github.com/Mipanox/Bird_cocktail/tree/master/experiments/AWS-experiments/multi-label/resnet_01) | BCE  | 0.770 | **0.854**| 0.741 | L2 | Training fastest |
| [DenseNet](https://github.com/Mipanox/Bird_cocktail/tree/master/experiments/AWS-experiments/multi-label/densenet_07) | BCE | 0.703 | 0.729 | 0.734 | Early stopping | high variance |

The conclusions to draw from the various models are left to the users.

_[(back to top)](https://github.com/Mipanox/Bird_cocktail#content)_

---
## 5. Reproducing Our Results
### Download Our Datasets
The datasets (pre-processed spectrograms) used in generating results shown in this README can be found [here](https://drive.google.com/open?id=1cBIc1meaP2Aj1CRFIGB-ZHB9stoJN6eO). They were converted from > 25 hours of `mp3` recordings from xeno-canto database and consist of > 40,000 pre-processed spectrograms and numerous noise samples.

### Using the Pre-trained Models
Download the hyperparameters and the best/last models [here](https://drive.google.com/open?id=1g61n2wXKkL3DQQ4_9Wf1YXSwqnkXojeo)

There are two ways of using them: 

(1) As initial weights: It is possible to re-train the model from a given status of weights. We recommend renaming the `pth.tar` file of your selection to prevent overwritting it with your own model run. For instance, to initialize training from `renamed_best.pth.tar`, which should reside in the same folder, run:
```
python train.py --data_dir <path_to_splitted_datasets> --model_dir <path_to_this_model_folder> --restore_file renamed_best
```

(2) For evaluation on test set (can omit the `--restore_file *` argument if using the `best.pth.rar`):
```
python evaluate.py --data_dir <path-to-test-data> --model_dir <path-to-model-folder> --restore_file some_model
```
where there is supposed to be a file called `some_model.pth.rar`

_[(back to top)](https://github.com/Mipanox/Bird_cocktail#content)_

---
## 6. Epilogue
### Future Perspectives
Given that we can only do 10 species at this point, which is not pratical for real-life situations, we want to scale up the model to at least 100 species (i.e. multi-class multi-label classification for 100 species, which is a reasonable number for any given local region). On the other hand, it is suggested that we can compare our models to some baseline models, e.g. traditional machine learning models, to have a better feeling of how far have we exceeded "so-so" performance.

Finally, the authors would like to thank Amazon Web Services and the Stanford CS230, 2018 Winter course teaching staff for their generous support of computing resources and guidelines for project, as well as the extremely helpful example codes and tutorials.

_[(back to top)](https://github.com/Mipanox/Bird_cocktail#content)_

---
## 7. References
### Papers
[p1] Huang _et al._, Densely Connected Convolutional Networks. (2016). arXiv: [1608.06993](https://arxiv.org/abs/1608.06993)

[p2] He _et al._, Deep Residual Learning for Image Recognition. (2015). arXiv: [1512.03385](https://arxiv.org/abs/1512.03385)

[p3] Li _et al._, Improving Pairwise Ranking for Multi-label Image Classification. (2017). arXiv: [1704.03135](https://arxiv.org/abs/1704.03135)

[p4] Zhang _et al._, Binary relevance for multi-label learning: an overview. ([2017](https://doi.org/10.1007/s11704-017-7031-7)). Frontiers of Computer Science, pp. 1-12

[p5] Szegedy _et al._, Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning. (2016). arXiv: [1602.07261](https://arxiv.org/abs/1602.07261)

[p6] Iandole _et al._, SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size. (2016) .arXiv: [1602.07360](https://arxiv.org/abs/1602.07360)

[p7] Gong _et al._, Deep Convolutional Ranking for Multilabel Image Annotation. (2013). arXiv: [1312.4894](https://arxiv.org/abs/1312.4894)

### Repositories
[r1] Stanford CS230 Example Code [Repository](https://github.com/cs230-stanford/cs230-code-examples/tree/master/pytorch/vision)

[r2] Kahl _et al._, [Working Notes](http://ceur-ws.org/Vol-1866/paper_143.pdf) of BirdCLEF. (2017). Github [repository](https://github.com/kahst/BirdCLEF2017)

[r3] Amos and Kolter, A PyTorch Implementation of DenseNet. Github [repository](https://github.com/bamos/densenet.pytorch); also andreaazzini's [repository](https://github.com/andreaazzini/multidensenet)

[r4] Tensorflow Model Zoo for Torch7 and PyTorch. Github [repository](https://github.com/Cadene/tensorflow-model-zoo.torch)

### Others
[o1] e.g. [BirdCLEF](http://www.imageclef.org/lifeclef/2017/bird), [EADM](http://sabiod.univ-tln.fr/EADM/), [BirdGenie](http://www.birdgenie.com), [Bird Song Id USA](http://us.isoperlaapps.com/BirdSongIdUSA.html); and proceedings/publications therein

[o2] [xeno-canto](https://www.xeno-canto.org) database of bird sounds

[o3] Official PyTorch [tutorials](http://pytorch.org/tutorials/), [documentation](http://pytorch.org/docs/0.3.1/), and [source code](https://github.com/pytorch/pytorch)

