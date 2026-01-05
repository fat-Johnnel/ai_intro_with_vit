# ai导论课设

基于ljn的导论，这一版使用的是自注意力机制的ViT算法，用于图像识别。导论报告可以基于两种算法的原理和效果对比等展开。

## 数据集

数据集下载： https://www.microsoft.com/en-us/download/details.aspx?id=54765

下载解压后放到程序根目录即可

## 依赖

依赖使用uv管理，如果不想用uv的话可以自己开venv，运行以下即可

```bash
pip install wandb
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

## 训练数据

基于数据集，将数据集进行切分预处理即可

## 模型权重

artifacts文件夹中

## 运行

```bash
python main.py <command> <args>
```

### 数据集切分 prepare-data

`--raw-train-dir` 存放原始数据集的路径（比如`PetImages`）

`--val-ratio` 评估集比例

`--test-ratio` 测试集比例

使用vit算法需要将图像转换成224\*224或者384\*384的正方形图片

`--img-size`: 若指定，则对每张图 pad 为正方形后 resize 到该尺寸；如果不想改变尺寸，传`--img-size 0` 或不使用该参数（默认会处理为 224）。

`--pad-color`: 填充颜色，格式 R,G,B（0-255），默认黑色 0,0,0。

`--dedupe`: 若启用，会基于 sha256 去重。

### 训练 train

`--data-dir` 切分后的数据集位置

`--epochs` `--lr` `--weight-decay` `--dropout` 超参数

`--wandb` 启用wandb

如果需要使用wandb监控，可以去 https://wandb.ai 注册账户后跟着他的教程登录

### 预测 predict

`--checkpoint` 模型路径，如`artifacts/model_best.pt`

`--image` 需要分类的图像路径

### 评估 eval

`--checkpoint` 模型路径，如`artifacts/model_best.pt`

`--data-dir` 切分后的数据集位置