# Chinese Font Recoginition

Chinese font recognition using a pretrained network - VGG16.

### Download VGGNet

This model will be using a pretrained network from https://github.com/machrisaa/tensorflow-vgg. Make sure to clone this repository to the directory you're working from. You also need to rename it so it has an underscore instead of a dash.

> git clone https://github.com/machrisaa/tensorflow-vgg.git tensorflow_vgg

Then you should run `download_VGGNet.py` to download the parameter file of VGGNet.

### Download Dataset

I have created a chinese font dataset which contains 5 kinds of fonts:

* songti	- 宋体
* fangsong	- 仿宋
* heiti		- 黑体
* lishu		- 隶书
* kaiti		- 楷体

You can download the dataset from https://pan.baidu.com/s/1pLuOuLH, then unzip the file running follow command in terminal and get the dataset. Make sure to unzip this file to the directory youre working from.

> tar -xzvf font_img.tar.gz

The images of fonts should be in the `./data/img/` directory.

### Get Codes of VGGNet

Now you can run through all the images dataset and get codes of VGGNet for each of them using `get_vgg_codes.py`.

Codes and corresponding lables will be written to `vgg_codes` and `labels` files.

### Train the Model

Run `model.py` to train the model.
