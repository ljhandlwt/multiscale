# multiscale
multiscale inception-v3 reid market-1501
## 运行train代码的方法：
- 在multiscale文件夹下运行sh ./scripts/train_multiscale_on_market-1501.sh 与 sh ./scripts/train_inceptionV3_299_on_market-1501.sh 分别对应multiscale和singlescale的299*299规格


## 运行get_feature代码的方法：
- 在multiscale文件夹下运行sh ./scripts/get_probe_features.sh, sh ./scripts/get_gallery_features.sh, 
./scripts/get_probe_features_single.sh, sh ./scripts/get_gallery_features_single.sh
分别对应multiscale的probe特征，gallery特征； singlescale的probe特征，gallery特征。
运行后会在multiscale文件夹下产生对应的mat文件，但multiscale和singlescale的feature名字是一样的，所以请不要同时运行。

## 运行test代码的方法：
- 在产生了所有的6个mat文件后，在multiscale文件夹下运行sh eval.sh
<<<<<<< HEAD

batch_size调成16 √
dropout调到0.5 √

check一下convert，数据的path是否对应
改一下loss，使用定义好的接口
50000，16
开分支：bn的scale改成true
图片输入网络大小改为：256 * 128
不同学习器：adam sgd
adam learning rate: 2e-4 5e-5 1e-5
sgd learning rate: 1e-2 1e-3 1e-4

batch_size:16
acc: 26.7

dropout: 0.5
acc: 30

bn的scale改成true
acc: 31

sgd, acc :30
adam, acc: 
=======
>>>>>>> b968fa541bd765703b4be8a6e90bed47adbdb46b
