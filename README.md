Convolution Neural Network
===
This is a MNIST number recognition problem solved by a convolution neural network model. <br />
Implement by tensorflow 0.11
![MNIST](https://github.com/m516825/CNN-MNIST/blob/master/img.png)
<br/>
<br/>
## Requirement
```
python2.7
tensorflow 0.11
progressbar
numpy
``` 

## Usage

Download MNIST training data and its labels from [here](http://yann.lecun.com/exdb/mnist/) <br/>

train-images-idx3-ubyte.gz:  training set images (9912422 bytes) <br/>
train-labels-idx1-ubyte.gz:  training set labels (28881 bytes) <br/>

Place "<em>train-images-idx3-ubyte</em>" and "<em>train-labels-idx1-ubyte</em>" in the same directory as this project

```
$ python cnn.py
```

##Performance

Train/Dev: 60000/2000

| Data | Accuarrcy |
| :---: |:---:|
| train | 0.998 |
| dev | 0.995 |

## Prediction
This code will automatically dump the prediction output of the <em>test-image</em>
<br/>
<br/>
<br/>
TODO: image distortion
