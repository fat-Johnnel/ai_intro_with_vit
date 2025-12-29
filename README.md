# ai导论课设

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

https://api.wandb.ai/links/paulkm-huazhong-university-of-science-and-technology/ok919efq

## 模型权重

https://huggingface.co/paulkm/ai_intro_curriculum_design/tree/main/artifacts

下载后放到artifacts文件夹中即可

## 运行

```bash
python main.py <command> <args>
```

### 数据集切分 prepare-data

`--raw-train-dir` 存放原始数据集的路径（比如`PetImages`）

`--val-ratio` 评估集比例

`--test-ratio` 测试集比例

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