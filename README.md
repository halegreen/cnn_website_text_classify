# cnn_website_text_classify
使用CNN对网站文本进行分类，基于tensorflow， 具体实现说明参见[使用CNN进行网站文本分类](https://zoeshaw101.github.io/2017/09/03/%E4%BD%BF%E7%94%A8CNN%E8%BF%9B%E8%A1%8C%E7%BD%91%E7%AB%99%E6%96%87%E6%9C%AC%E5%88%86%E7%B1%BB/)

## 文件结构
/---
|---- data_helper.py  : 读取训练数据，包括文本清洗、进行文本句子补齐（sentence padding)等预处理。
|---- word2vec_helpers.py : 进行word2vec向量化，主要借助gensim库，并将训练好的word2vec模型保存在run/目录下。
|---- text_cnn.py : 定义了一个类用来描述网络结构：一个卷积层加一个池化层。
|---- mytrain.py : 训练模型，包括超参数定义、计算图的描述。
|---- eval_helper.py :  读取需要进行预测的真实数据，以及进行数据check。
|---- eval.py : 使用训练好的模型进行预测真实数据。

## 使用方法
- 训练模型 ：
```
> python mytrain.py
```

- 预测真实数据：
```
python eval.py -checkfile_dir = {your_code_path/runs/checkfile}

## 实验结果
在训练和验证集上表现良好，正确率达95%左右；在真实数据集（无标签）上表现欠佳。
