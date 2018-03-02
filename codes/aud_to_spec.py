###############################################
# Converting wav audio files to spectrograms  #
#  Partly borrowed from Stefan Kahl's         #
#  implementation for BirdCLEF2017 challenge: #
#   https://github.com/kahst/BirdCLEF2017/    #
###############################################

from __future__ import print_function
import os
import traceback
import numpy as np
import cv2
import librosa
from tqdm import tqdm
import scipy.ndimage as ndimage
from PIL import Image
from optparse import OptionParser

parser = OptionParser()
parser.add_option("--src_dir"  , dest="src_dir")
parser.add_option("--spec_dir" , dest="spec_dir")
parser.add_option("--threshold", dest="threshold", default=14.)
(options, args) = parser.parse_args()

## paths
src_dir  = options.src_dir
spec_dir = options.spec_dir

## specify maximum number of spectrograms per species (-1 = No limit)
MAX_SPECS = -1

## limit number of species? (None = No limit)
MAX_SPECIES = None

## default threshold
threshold = options.threshold

######################################################

def getSpec(sig, rate, power=2, **kwargs):
    """
    Get spectrogram from signal split
    
    Options:
    - power : integer, 1 or 2
      Specify the power of spectrogram. Defaults to intensity
      For power = 1, return amplitude spectrogram
    """
    
    ## melspetrogram
    magspec = librosa.feature.melspectrogram(y=sig,sr=rate,power=power,**kwargs)
    
    return magspec


def getMultiSpec(path, seconds=3., overlap=2.5, minlen=3., **kwargs):
    """
    Split signal into chunks with overlap of certain seconds and tunable minimum length
    
    Options:
    - seconds : np.float
      Length of output chunk (segment) of audio. Defaults to 3 seconds
    
    - overlap : np.float
      Length of overlapping window of segmentation. Defaults to 2.5 seconds
      
    - minlen : np.float
      Minimal length of output spectrogram in seconds. Defaults to 3 seconds.
    """
    #--
    ## open audio file
    sig, rate = librosa.load(path,sr=None) # natural sample rate
    
    ## adjust to different sample rates
    if rate != 44100:
        sig  = librosa.resample(sig, rate, 44100)

    #--
    ## split signal with overlap
    sig_splits = []
    for i in xrange(0, len(sig), int((seconds - overlap) * rate)):
        split = sig[i:i + int(seconds*rate)]
        if len(split) >= minlen * rate:
            sig_splits.append(split)

    ## if signal too short for segmentation, append it anyway
    if len(sig_splits) == 0:
        sig_splits.append(sig)
    
    #--
    ## calculate spectrogram for each split
    for sig in sig_splits:
        magspec = getSpec(sig, rate, **kwargs)

        yield magspec


def hasBird(spec, raw_thresh=10., threshold=threshold):
    """
    Decide if given spectrum shows bird sounds or noise only
    
    Options:
    - raw_thresh : float
      Threshold for the absolute value of signal before fancy processing
      Defaults to 10
    
    - threshold : float
      Threshold value for separating real signal / noise. Defaults to 30.
      Changeable via command-line option (see above)
    """
    ## working copy
    img = spec.copy()
    
    ### STEP 0: get rid of highest/lowest freq bins
    #-- total of 128 bins, to integer vals due to JPEG format
    img = img[20:100,:].astype(int).astype(np.float32)
    
    bird, rthresh = False, 0. # if too weak, treat as no signal no matter what
    if img.max() > raw_thresh: # absolute value of maximum signal
        ### STEP 1: normalize to [0,1] for processing
        img -= img.min()
        img /= img.max()
    
        ### STEP 2: Median blur
        img = cv2.medianBlur(img,5)

        ### STEP 3: Median threshold
        col_median = np.median(img, axis=0, keepdims=True)
        row_median = np.median(img, axis=1, keepdims=True)

        img[img < row_median * 3] = 0.
        img[img < col_median * 4] = 0.
        img[img > 0.] = 1.

        ### STEP 4: Morph Closing
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((5,5), np.float32))
    
        ### STEP 5: Count columns and rows with signal
        #-- (Note: We only use rows with signal as threshold -- time axis for fixed freqs)
    
        ##### row has signal?
        row_max = np.max(img, axis=1)
        row_max = ndimage.morphology.binary_dilation(row_max, iterations=2).astype(row_max.dtype)
        rthresh = row_max.sum()
    
        ### STEP 6: Apply threshold
        if rthresh >= threshold:
            bird = True
    
    return bird, rthresh

######################################################

if __name__ == "__main__":
    # list all bird species
    birds = [src_dir + bird + '/' for bird in sorted(os.listdir(src_dir))][:MAX_SPECIES]
    print("# of bird species:{}".format(len(birds)))

    # parse bird species
    for bird in birds:
        print('Now processing : {}'.format(bird))
        tot_specs = 0
        
        ## get all audio files - can handle wav and mp3
        aud_files = [bird + aud for aud in sorted(os.listdir(bird))]
        
        ## parse audio files
        t = tqdm(aud_files)
        for aud in t:
            spec_cnt = 0
                
            try:
                ## get every spec from each audio file
                for spec in getMultiSpec(aud):
                    ## does spec contain bird sounds?
                    isbird, thresh = hasBird(spec)
                    
                    ## new target path -> rejected specs will be copied to "noise" folder
                    if isbird:
                        dst_dir = spec_dir + bird.split("/")[-2] + "/"
                    else:
                        dst_dir = spec_dir + "noise/" + bird.split("/")[-2] + "/"
                    
                    ## make target dir
                    if not os.path.exists(dst_dir):
                        os.makedirs(dst_dir)
                    
                    if spec.shape[1] > 192: # for cropping. specified minlen only assures len in sec not sampling
                        ## write spec to target dir
                        file_path = dst_dir+str(thresh) + "_" + aud.split("/")[-1].rsplit(".")[0] + \
                                                          "_" + str(spec_cnt) + ".jpg"
                        ### PIL prevents auto-rescaling to [0,255]
                        img = Image.fromarray(spec)
                        img = img.convert('L')          # to grey scale
                        img.save(file_path, quality=95) # minimal distortion
                    
                        spec_cnt  += 1
                        tot_specs += 1

                ## exceeds spec limit?
                if tot_specs >= MAX_SPECS and MAX_SPECS > -1:
                    break
                    
                t.set_description("Spectrogram Count : {}".format(spec_cnt))
                    
            except:
                tqdm.write("----ERROR----: file{}".format(aud))
                traceback.print_exc()
                pass    