# Sphereface-Prune
Based on Caffe, Caffe's Python interface, the code realizes the compression and acceleration of  face recognition model Sphereface. 
 
## Requirement
Caffe 1.0.0
Python 2.7.6
Caffe's interface for Python
Matplotlib 1.3.1
Numpy 1.13.1

## Sphereface

### Structure
![enter image description here](http://wx4.sinaimg.cn/mw690/710b0c10ly1fx0g2yfozxj210z0dfmzz.jpg)

### Train
Trained by CASIA-Webface dataset
![enter image description here](http://wx4.sinaimg.cn/mw690/710b0c10ly1fx0g5rv3tbj20la0k141n.jpg)

### Test accuracy
10-folds cross validation by LFW dataset
![enter image description here](http://wx2.sinaimg.cn/mw690/710b0c10ly1fx0g8ui3lkj20ru06jjss.jpg)

## Prune
### Sphereface-4
![enter image description here](http://wx3.sinaimg.cn/mw690/710b0c10ly1fx0gdd3xawj20iq0lnwft.jpg)
### Sphereface-10
Deal with the residual block in Sphereface-10
![enter image description here](http://wx4.sinaimg.cn/mw690/710b0c10ly1fx0gynalchj20ax0j3t9j.jpg)

### Comparison
![enter image description here](http://wx3.sinaimg.cn/mw690/710b0c10ly1fx0gkur079j21050b1tbw.jpg)


## References
[SphereFace: Deep Hypersphere Embedding for Face Recognition][1]
[ThiNet: A Filter Level Pruning Method for Deep Neural Network Compression][2]

[SphereFace: Deep Hypersphere Embedding for Face Recognition] (https://arxiv.org/abs/1704.08063 )
[ThiNet: A Filter Level Pruning Method for Deep Neural Network Compression] (https://arxiv.org/abs/1707.06342)

