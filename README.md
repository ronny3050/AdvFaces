# AdvFaces: Adversarial Face Synthesis
By Debayan Deb, Jianbang Zhang, and Anil K. Jain

<a href="https://arxiv.org/abs/1908.05008"><img src="https://raw.githubusercontent.com/ronny3050/AdvFaces/master/assets/cover.png" width="50%"></a>

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/2/2d/Tensorflow_logo.svg/1000px-Tensorflow_logo.svg.png" align="right" width="100"/>

A tensorflow implementation of [AdvFaces](https://arxiv.org/abs/1811.10100), a fully automatic adversarial face generatorthat learns to generate minimal perturbations in the salient facial regions via Generative Adversarial Networks. Once AdvFaces is trained, it can automatically generate imperceptible perturbations that can fool state-of-the-art face matchers with attack success rates as high as 97.22% and 24.30% for obfuscation and impersonation attacks, respectively.

## <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/2/2d/Tensorflow_logo.svg/1000px-Tensorflow_logo.svg.png" width="25"/> Tensorflow release
Currently this repo is compatible with Tensorflow r1.14.0.

## <img src="https://image.flaticon.com/icons/svg/182/182321.svg" width="25"/> Citation

    @article{deb2019advfaces,
    title={Advfaces: Adversarial face synthesis},
    author={Deb, Debayan and Zhang, Jianbang and Jain, Anil K},
    journal={arXiv preprint arXiv:1908.05008},
    year={2019}}

## <img src="https://image.flaticon.com/icons/svg/1/1383.svg" width="25"/> Usage
### Training
1. The configuration files for training are saved under ```config/``` folder, where you can define the dataset prefix, training list, model file, attack setting (obfuscation or impersonation) and other hyper-parameters. Use the following command to run the default training configuration:
    ``` Shell
    python train.py config/default.py
    ```
    The command will create an folder under ```log/default/``` which saves all the checkpoints, test samples and summaries. The model directory is named as the time you start training.

### Testing
* Run the test code in the following format:
    ```Shell
    python test_adversaries.py
    ```
* For example, if you want to use the pre-trained model, download the model and unzip it into ```pretrained``` folder. Then, run 
    ```Shell
    python test_adversaries.py
    ```

## <img src="https://image.flaticon.com/icons/svg/48/48541.svg" width="25"/> Pre-trained Model
##### OBFUSCATION MODEL: 
[Google Drive](https://drive.google.com/file/d/1QfptqO9WffhjUQmrNVYuSVF-iCVT_U5h/view?usp=sharing)
