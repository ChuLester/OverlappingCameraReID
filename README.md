# OverlappingCameraReID
Using MTMTC idea tracks Overlapping person. We also use human skeleton and person location to reinforce tracking accuracy.

System package env:
    python 3.6
    tensorflow 1.15.1
    sklearn
    OpenPose-tf
    Opencv 4.0.0
    Keras

Model please download to your disk
ReID model to ./model - <https://drive.google.com/drive/folders/1w3HsyxnANcXOudwEWP0vGWtpDkc6iM-4?usp=sharing>
OpenPose model to ./models -<https://drive.google.com/drive/folders/1VOzuQWm-iy4Zw-YeIgkM6wpGi9zSSicf?usp=sharing>

Video please download to your disk
VideoDataSet to ./VideoDataSet -<https://drive.google.com/drive/folders/1R_XRNnYDSagS8VKeC8kDkr7JS-NYuipN?usp=sharing>

This program use EPFL Lab 4 to perform our algorithm. 
If you want to test your DataSet. First, you need to Use demo_nopos.py make track file.(Save Varible T)
Second, Use ./utils/For_WorldPosition.py make MDS_map file to ./MDS_pos

If you don't like use our reid model, you will use yourself person's dataset to train your model.
We prepare training model in utils dir.
    MakeTripletSample.py : Make Triplet with person's dataset(the dataset which ready classify)
    Pretrain_AMSoftmax.py : Make preTrain model weight to Tripletloss_Resnet.py
    Tripletloss_Resnet.py : final to fit reid model to use.

If you like our program or you have any problem, contact me: <dxxlin2065@gmail.com>