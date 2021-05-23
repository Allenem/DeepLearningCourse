# 国科大计算机学院深度学习作业

## CONTENT

- hw0：文献阅读（10篇）
- hw1：MNIST数据集手写数字识别（准确率98.83%）
- hw2：猫狗二分类（准确率91%以上）
- hw3：自动写诗（首句写诗，藏头写诗）
- hw4：电影中文影评情感分类（准确率，精度，召回率，F1分数，混淆矩阵指标分别达到了：86.18%，0.8626，0.8579，0.8603，[[157.0, 26.0], [25.0, 161.0]]）

## 文件列表结构

由于猫狗分类、维基百科中文word2vec.bin等数据文件较大，本人并未全部上传至此仓库，可在如下地址下载数据集。

- 实验1数据：[MINIST](http://yann.lecun.com/exdb/mnist/) 手写数据集
- 实验2数据：从[kaggle比赛官网](https://www.kaggle.com/c/dogs-vs-cats/data) 下载所需的猫狗分类数据；或者直接从此下载[训练集](https://yun.sfo2.digitaloceanspaces.com/pytorch_book/pytorch_book/data/dogcat/train.zip)和[测试集](https://yun.sfo2.digitaloceanspaces.com/pytorch_book/pytorch_book/data/dogcat/test1.zip)
- 实验3数据：`tang.npz` 文件较小，已上传至本仓库，或者：链接：https://pan.baidu.com/s/1dtf9HOEY1jzqR51tyyQfjg 提取码：65qv 复制这段内容后打开百度网盘手机App，操作更方便哦
- 实验4数据：`Dataset` (包含`test.txt`, `train.txt`, `validation.txt`, `wiki_word2vec_50.bin`)：链接：https://pan.baidu.com/s/1VDYXwjSLO1sTC0XJKq9ggA 
提取码：buuq 复制这段内容后打开百度网盘手机App，操作更方便哦
            
            
            
            

```
.
│  README.md
│
├─hw0_paper_reading
│      0Deep residual learning for image recognition.pdf
│      0图像识别的深度残差学习.pptx
│      1Long-term Recurrent Convolutional Networks for Visual Recognition and Description.pdf
│      1用于视觉识别和描述的长期循环卷积网络.pptx
│      2Speech recognition with deep recurrent neural networks.pdf
│      2基于深度循环神经网络的语音识别.pptx
│      3Attention Is All You Need.pdf
│      3你只需要注意力.pptx
│      4A Reinforced Topic-Aware Convolutional Sequence-to-Sequence Model for Abstractive Text Summarization.pdf
│      4一种增强主题感知的卷积序列对序列的摘要模型.pptx
│      5Adaptive RNN for target-dependent Twitter Sentiment Classification.pdf
│      5自适应递归神经网络用于目标相关的推特情感分类.pptx
│      6Convolutional Neural Networks for Sentence Classification.pdf
│      6用于句子分类的卷积神经网络.pptx
│      7Fast R-CNN.pdf
│      7快速区域卷积神经网络.pptx
│      8You Only Look Once-Unified, Real-Time Object Detection.pdf
│      8你只需看一次——统一的实时的目标检测.pptx
│      9YOLOv3- An Incremental Improvement.pdf
│      9YOLOv3——一个渐进的改进.pptx
│
├─hw1_writing_digit_recognition
│  │  index.py
│  │  OUTPUT.txt
│  │  实验报告.docx
│  │  实验报告.pptx
│  │
│  └─MNIST_data
│          t10k-images-idx3-ubyte.gz
│          t10k-labels-idx1-ubyte.gz
│          train-images-idx3-ubyte.gz
│          train-labels-idx1-ubyte.gz
│
├─hw2_catVSdog_classification
│  │  index.py
│  │  OUTPUT.txt
│  │  实验报告.docx
│  │  实验报告.pptx
│  │
│  └─data
│      ├─test
│      │  ├─cat
│      │  │      cat.12000.jpg
│      │  │      ...
│      │  │      cat.12499.jpg
│      │  │
│      │  └─dog
│      │          dog.12000.jpg
│      │          ...
│      │          dog.12499.jpg
│      │
│      └─train
│          ├─cat
│          │      cat.0.jpg
│          │      ...
│          │      cat.999.jpg
│          │
│          └─dog
│                  dog.0.jpg
│                  ...
│                  dog.999.jpg
│
├─hw3_write_poetry_automatically
│      aotomatic_writing_poetry.ipynb
│      model.pth
│      tang.npz
│      实验报告.docx
│      实验报告.pptx
│
└─hw4_movie_chinese_comments_sentiment_classification
    │  Chinese_movie_comments_sentiment_classification.ipynb
    │  model.pth
    │  实验报告.docx
    │  实验报告.pptx
    │
    └─Dataset
            test.txt
            train.txt
            validation.txt
            wiki_word2vec_50.bin
```