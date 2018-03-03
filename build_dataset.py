"""
 Split dataset into train/val/test sets
 - Largely inherited from Stanford CS230 example code:
   https://github.com/cs230-stanford/cs230-code-examples/tree/master/pytorch/vision

 The dataset comes into the following format:
    train/
        species001/
            spec0001.jpg
        ...
    val/
        species001/
            spec0001.jpg
        ...
    test/
        species001/
            spec0001.jpg
        ...
    noise/

 For our purpose, data augmentation (if any) will be done
 when the training process calls the dataloader. 
 In here, we simply split the pre-processed spectrograms for all species.

 User can specify the percentage of the train/val/test split,
 which depends on the number of species we want to train,
 which comes as another optional input argument.

 By default, we train 200 species with 98%/1%/1% split
"""

from __future__ import print_function
import argparse
import random
import os
from shutil import copy as cp

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir'  , help="Directory with the pre-processed dataset")
parser.add_argument('--output_dir', help="Directory to write train/val/test split")
parser.add_argument('--train_per' , help="Percentage of training set", default=98, type=float)
parser.add_argument('--max_spec'  , help="Maximum number of species for the model", default=200, type=int)


#####################
## Helper function ##
def dir_size(path):
    """ Size of a folder's content """
    return sum(os.path.getsize(os.path.join(args.data_dir,path,f)) \
               for f in os.listdir(os.path.join(args.data_dir,path)))


#----------------------------
if __name__ == '__main__':
    args = parser.parse_args()

    assert os.path.isdir(args.data_dir), "Couldn't find the dataset at {}".format(args.data_dir)

    # Get the filenames in each directory
    filenames = []
    ## sort by "richness" of each species    
    birdnames = [args.data_dir + bird + '/' for bird in \
                 sorted(os.listdir(args.data_dir),key=dir_size,reverse=True) if bird != 'noise'] 
    for i,bird in enumerate(birdnames):
        if i >= args.max_spec: break ## only use first max_spec species 
        filenames += [os.path.join(bird, f) for f in os.listdir(bird) if f.endswith('.jpg')]

    # Split dataset
    ## Make sure to always shuffle with a fixed seed so that the split is reproducible
    random.seed(1240)
    filenames.sort()
    random.shuffle(filenames)

    split = int(args.train_per/100. * len(filenames))
    train_filenames = filenames[:split]
    val_filenames   = filenames[split:(len(filenames)+split)/2]
    test_filenames  = filenames[(len(filenames)+split)/2:]

    filenames = {'train': train_filenames,
                 'val'  : val_filenames,
                 'test' : test_filenames}

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    else:
        print("Warning: output dir {} already exists".format(args.output_dir))

    # Preprocess train, val and test
    for split in ['train', 'val', 'test']:
        output_dir_split = os.path.join(args.output_dir, '{}'.format(split))
        if not os.path.exists(output_dir_split):
            os.mkdir(output_dir_split)
        else:
            print("Warning: dir {} already exists".format(output_dir_split))

        print("Copying preprocessed data to {} ...".format(split, output_dir_split))
        for filename in tqdm(filenames[split]):
            spepath = os.path.join(output_dir_split, '{}'.format(filename.split("/")[-2]))
            if not os.path.exists(spepath):
                os.mkdir(spepath)
            cp(filename, spepath)

    print("Done building dataset")
