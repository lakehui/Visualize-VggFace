# Visualize-VggFace

# Requirment
caffe and build with "WITH_PYTHON_LAYER := 1" 

# Experiment
MtripLoss are added to conv4_3, conv5_3, fc6 and fc7, and the MEucliLoss is added to conv1_1
![image1](https://github.com/lakehui/Visualize-VggFace/tree/master/VGGFacenet/model/003.jpg)
![image2](https://github.com/lakehui/Visualize-VggFace/tree/master/VGGFacenet/model/backg.jpg)
![image3](https://github.com/lakehui/Visualize-VggFace/tree/master/VGGFacenet/back_pe.jpg)

This is a strange phenomenon. Using the 50% of larger response node to reconstruct input image, 
roughly the content of the input image can be restored. But when usin the 20-30% of smaller response 
node to reconstruct the input image, the roughly the content of the input image also can be restored.
![image4](https://github.com/lakehui/Visualize-VggFace/tree/master/VGGFacenet/reconst_08.jpg)
![image5](https://github.com/lakehui/Visualize-VggFace/tree/master/VGGFacenet/reconst_03.jpg)
![image6](https://github.com/lakehui/Visualize-VggFace/tree/master/VGGFacenet/reconstall.jpg)
![image7](https://github.com/lakehui/Visualize-VggFace/tree/master/VGGFacenet/reconst_09.jpg)
