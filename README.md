# PTBXL_Augmentaton

## 所需环境

需要注意的是PyTorch，按照自己显卡型号装好就行，其他的先运行代码。缺啥再装啥

## 运行步骤

1、运行 xlsx2npy.py，将 PTBXL 文件夹中的xlsx文件转换为 npy 文件， 生成 npy_files 文件夹

2、运行 split_npy_files.py，将 npy_files 文件夹按照 1：2 划分为测试和训练集，用于训练 mae，生成 mae_train 和 mae_test 文件夹

3、运行 train_mae_mask.py，训练 mae模型

4、运行  [mae_evaluate.py](mae_evaluate.py) ，评估 mae效果

5、运行 signal_expand.py， 用于对信号进行扩充，每个类别扩充到3000条，生成 mae_add 文件夹

6、运行  [bulid_dataset.py](bulid_dataset.py) ，用于将  [npy_files](npy_files) 文件夹进行划分，生成 data 文件夹

7、运行  [train_classification.py](train_classification.py) ，用于对未信号扩充前的数据进行训练

8、运行  [predict_classification.py](predict_classification.py) ，用于对未信号扩充前的数据进行测试

9、运行  [bulid_dataset_add.py](bulid_dataset_add.py) ，用于对 [npy_files](npy_files) 和 mae_add 文件夹中的 npy文件进行划分， 生成 data_add 文件夹

10、运行  [train_classification_add.py](train_classification_add.py) ，用于对未信号扩充后的数据进行训练

11、运行  [predict_classification_add.py](predict_classification_add.py) ，用于对未信号扩充后的数据进行测试

