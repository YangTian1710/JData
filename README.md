# JData 2018大数据实训py文件说明
- MyFeature.py：整理给定的train和test_b数据，得到train_featureVB.csv和test_featureVB.csv文件
- MyModel(0.75).py:需要先运行MyFeature.py得到train_featureVB.csv和test_featureVB.csv文件，利用lightgbm模型只调了learning_rate一个参数，预测准确率为0.7583
- MyModel(0.77).py:利用lightgbm模型，配置了部分参数，训练时利用over_sample取平衡数据集进行训练，准确率为0.7740

