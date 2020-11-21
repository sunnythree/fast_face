### face_align
***this is a project for: Full convolution face alignment***  

#### show
![pic1](https://github.com/sunnythree/face_alignment/blob/master/data/t1.png)
![pic2](https://github.com/sunnythree/face_alignment/blob/master/data/t2.png)
![pic3](https://github.com/sunnythree/face_alignment/blob/master/data/t3.png)

#### description
CnnAlignHard is a better model for face alignment,so use XXXHard.py 

#### dataset
I trained the model(data/face_align_hard.pt) on helen dataset:  
[helen dataset](http://www.ifp.illinois.edu/~vuongle2/helen/)
if you want to train your own model on helen datset, please download the helen dataset   
and change you own helen dataset's path in Config.py.  
so:  
* you need download annatations and Train/Test images.
* write all train/test images file name to train.txt/test.txt file:  
***cd Train***  
***ls > train.txt***   
***mv train.txt ..***    
***cd Test***  
***ls > test.txt***  
***mv test.txt***   
The dataset directory like this:  
![dataset](https://github.com/sunnythree/face_alignment/blob/master/data/dic.png)  
#### train  
```
python3.6 TrainHard.py -b 100 -l 0.001 -e 2000 -s 500  
```
if you want train from last time, just add -p True
```
python3.6 TrainHard.py -b 100 -l 0.001 -e 2000 -s 500  -P true
```
#### test on  eval dataset
```
python3.6 TestHard.py
```
#### test with camera
```
python3.6 CameraShowHard.py
```