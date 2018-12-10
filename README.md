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
![enter image description here](https://github.com/isthatyoung/Sphereface-prune/blob/master/images/figure1.png)

### Train
Trained by CASIA-Webface dataset
![enter image description here](https://github.com/isthatyoung/Sphereface-prune/blob/master/images/figure2.png)

### Test accuracy
10-folds cross validation by LFW dataset
![enter image description here](https://github.com/isthatyoung/Sphereface-prune/blob/master/images/figure3.png)

## Prune
### Sphereface-4
![enter image description here](https://github.com/isthatyoung/Sphereface-prune/blob/master/images/figure4.png)
### Sphereface-10
#### Deal with the residual block in Sphereface-10
![enter image description here](https://github.com/isthatyoung/Sphereface-prune/blob/master/images/figure5.png)
### Fine-tuning
1 epoch fine-tuning after pruning every layer. Becuase CASIA Webface contains 452,723 face images, so 1 epoch means 1768 times iteration by setting batch size to 256. Basic leaning rate is set to 1 × 10−3.  
After all layers are pruned, do 8 to 9 epochs fine-tuning, basic learning rate is 1 × 10−3.  
![enter image description here](https://github.com/isthatyoung/Sphereface-prune/blob/master/images/figure6a.png)
![enter image description here](https://github.com/isthatyoung/Sphereface-prune/blob/master/images/figure6b.png)
### Comparison
![enter image description here](https://github.com/isthatyoung/Sphereface-prune/blob/master/images/figure7.png)


## References
[SphereFace: Deep Hypersphere Embedding for Face Recognition](https://arxiv.org/abs/1704.08063)   
[ThiNet: A Filter Level Pruning Method for Deep Neural Network Compression](https://arxiv.org/abs/1707.06342)

