## Data
In the paper, [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) is utilized for training and testing. To start this work, make HDF5 files of the same length and place them in the correct path. The file structure is as follows:<br/>
>--data/<br/>
>>--trainset_1.h5<br/>
>>...<br/>
>>--trainset_n.h5<br/>
>>--train_files.txt<br/>
>>--validset_1.h5<br/>
>>...<br/>
>>--validset_n.h5<br/>
>>--valid_files.txt<br/>

## Environment
Python 3.8.10<br/>
CUDA 11.8<br/>
Torch 2.0.0<br/>
Torchvision 0.15.1<br/>
OpenCV 4.11.0.86<br/>
H5py 3.11.0<br/>
Spectral 0.23.1<br/>
Einops 0.8.1<br/>
Meshio 5.3.5<br/>

## Usage
1. Download this repository via git or download the [ZIP file](https://github.com/MaxtBIT/LearningRDOptics/archive/refs/heads/main.zip) manually.
```
git clone https://github.com/MaxtBIT/LearningRDOptics.git
```
2. Download the [pre-trained models](https://drive.google.com/drive/folders/1Qlis8xs5p8LsyA6q7Bugv99URkshkOuz?usp=sharing) if you need.
3. Make the datasets and place them in correct paths. Then, adjust the settings in **utils.py** according to your data.
4. Run the file **main.py** to train or test a model.