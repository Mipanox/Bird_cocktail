############################
# batch convert mp3 to wav #
############################

ppath='/scratch/users/jasonhc/bird_cocktail/datasets/Xeno-Canto/'

## clean up log
rm ffmpeg.log

for d in "${ppath}"mp3/*; do
  ## navigate all species folders
  if [ -d "$d" ]; then
    ## mkdir in wav folder
    wav_path=${ppath}wav/$(echo $d | rev | cut -d'/' -f-1 | rev)
    mkdir ${wav_path}

    for i in "$d/"*; do
      #echo "$d/"$(basename $i .mp3).wav
      ffmpeg -i $i "${wav_path}/"$(basename $i .mp3).wav &>> ffmpeg.log
    done
  fi
done
