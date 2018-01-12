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

testing