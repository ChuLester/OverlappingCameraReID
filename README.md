# OverlappingCameraReID
Using MTMTC idea tracks Overlapping person. We also use human skeleton and person location to reinforce tracking accuracy.

System package env:<br />
    python 3.6<br />
    tensorflow 1.15.1<br />
    sklearn<br />
    OpenPose-tf<br />
    Opencv 4.0.0<br />
    Keras<br />
<br />
Model please download to your disk<br />
ReID model to ./model - <https://drive.google.com/drive/folders/1w3HsyxnANcXOudwEWP0vGWtpDkc6iM-4?usp=sharing><br />
OpenPose model to ./models -<https://drive.google.com/drive/folders/1VOzuQWm-iy4Zw-YeIgkM6wpGi9zSSicf?usp=sharing><br />

Video please download to your disk<br />
VideoDataSet to ./VideoDataSet -<https://drive.google.com/drive/folders/1R_XRNnYDSagS8VKeC8kDkr7JS-NYuipN?usp=sharing><br />

This program use EPFL Lab 4 to perform our algorithm. <br />
If you want to test your DataSet. First, you need to Use demo_nopos.py make track file.(Save Varible T)<br />
Second, Use ./utils/For_WorldPosition.py make MDS_map file to ./MDS_pos<br />

If you don't like use our reid model, you will use yourself person's dataset to train your model.<br />
We prepare training model in utils dir.<br />
    MakeTripletSample.py : Make Triplet with person's dataset(the dataset which ready classify)<br />
    Pretrain_AMSoftmax.py : Make preTrain model weight to Tripletloss_Resnet.py<br />
    Tripletloss_Resnet.py : final to fit reid model to use.<br />

If you like our program or you have any problem, contact me: <dxxlin2065@gmail.com>
