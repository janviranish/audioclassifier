### Overview
MATLAB-based audio classifier that uses mel-frequency features and k-NN to categorize x classes of environmental sounds.

## Pre-requisites
Download audio files or record via phone app

Organize by category

Download MATLAB Audio & Statistics and Machine Learning Toolbox

## Technical
Extracts 64 mel‑frequency features from each audio clip

Trains k-NN classifier (k=5) with 80/20 train-test split

Reports classification accuracy

## Results
Depending on how distinct the classes of sounds are, the accuracy varies. 

I.e. very similar sounds could lead to slightly lower accuracy, and as high as 100% accuracy for distinct sounds like thunder, siren & traffic.

## Limitations
Requires MATLAB Audio Toolbox

Performance ultimatley depends on quality and variety of sounds (∴ have a good mix with enough diveristy in volume and type)
