import numpy as np  # 导入NumPy库，用于高效的多维数组操作和数学计算
import pandas as pd  # 导入Pandas库，用于数据分析和操作，特别是提供DataFrame对象
import pickle  # 导入pickle库，用于对象序列化和反序列化，即保存和加载Python对象
import random  # 导入random库，提供生成随机数的功能
import copy  # 导入copy库，用于复制Python对象，包括浅复制和深复制
import gc  # 导入gc库，用于手动触发Python的垃圾收集机制

# 从tokenizers库导入多个模块，这是一个快速的文本分词库
from tokenizers import decoders, models, normalizers, pre_tokenizers, processors, trainers, Tokenizer

from transformers import PreTrainedTokenizerFast  # 从transformers库导入PreTrainedTokenizerFast，用于快速加载和使用预训练的分词器

from datasets import Dataset  # 从datasets库导入Dataset类，用于处理和准备机器学习和NLP任务的数据集
from tqdm.auto import tqdm  # 从tqdm库导入tqdm，用于在循环中添加进度条，提高用户体验

import transformers  # 导入transformers库，提供大量预训练模型用于NLP任务
import os  # 导入os库，用于操作系统级别的接口，如文件路径操作、环境变量访问等
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer  # 从transformers导入模型和训练相关的类
import torch  # 导入PyTorch库，是一个广泛使用的深度学习框架
from transformers import AutoTokenizer  # 从transformers库导入AutoTokenizer，用于自动加载和使用预训练分词器

from sklearn.model_selection import StratifiedKFold  # 从sklearn导入StratifiedKFold，用于分层抽样的交叉验证
from sklearn.pipeline import Pipeline  # 从sklearn导入Pipeline，用于创建机器学习的处理流水线
from sklearn.decomposition import TruncatedSVD, NMF, LatentDirichletAllocation  # 从sklearn导入降维算法

from scipy.sparse import csr_matrix  # 从scipy导入csr_matrix，用于高效存储和操作稀疏矩阵
from sklearn.neighbors import NearestNeighbors  # 从sklearn导入NearestNeighbors，用于最近邻搜索

# 从sklearn.feature_extraction.text导入文本特征提取器
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# 从sklearn导入多个机器学习模型
from sklearn.naive_bayes import MultinomialNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier, StackingClassifier, ExtraTreesClassifier, RandomForestClassifier
from xgboost import XGBClassifier  # 导入XGBoost库，用于梯度提升决策树模型
from lightgbm import LGBMClassifier  # 导入LightGBM库，同样用于梯度提升决策树模型
from catboost import CatBoostClassifier  # 导入CatBoost库，是另一种梯度提升决策树模型

# 使用pandas的read_csv函数从指定路径读取测试集CSV文件，并将数据存储在DataFrame对象test中。
test = pd.read_csv('/kaggle/input/llm-detect-ai-generated-text/test_essays.csv')

# 读取提交样本格式的CSV文件，并将其存储在DataFrame对象submission中。
submission = pd.read_csv('/kaggle/input/llm-detect-ai-generated-text/sample_submission.csv')

# 从指定路径读取原始训练集CSV文件，并将数据存储在DataFrame对象origin中。
origin = pd.read_csv('/kaggle/input/llm-detect-ai-generated-text/train_essays.csv')

# 读取另一个训练集CSV文件，这次是从一个不同的路径，并且明确指定列分隔符为逗号（sep=','）。
# 并将数据存储在DataFrame对象train中。
train = pd.read_csv('/kaggle/input/daigt-v2-train-dataset/train_v2_drcat_02.csv', sep=',')

# 下一行代码被注释掉了，如果取消注释，它将筛选出train DataFrame中"RDizzl3_seven"列为True的行。
# train = train[train["RDizzl3_seven"] == True] 

# 从train DataFrame中删除"text"列的重复行。
train = train.drop_duplicates(subset=['text'])

# 重置train DataFrame的索引。drop=True参数避免将旧索引添加为列。
# inplace=True参数表示在原地修改DataFrame，而不是创建一个新的DataFrame。
train.reset_index(drop=True, inplace=True)

# 计算train DataFrame的行数，即获取训练集的样本数量。
x_num = len(train)

# 生成一个随机的索引列表，这个列表的长度等于train的行数，范围也是从0到train的行数（不包括上限）。
# 这个随机列表用于之后的重排操作。
x = random.sample(range(len(train)), x_num)

# 创建一个新的DataFrame，名为tmp。这个DataFrame是通过遍历上面生成的随机索引列表x，
# 并从train中按索引取行来构建的。这样，tmp中的行就是按照x中随机顺序重排的train的行。
tmp = pd.DataFrame([train.iloc[i] for i in x])

# 将重排后的DataFrame赋值给train，以便后续操作使用重排后的数据集。
train = tmp

# 重置train DataFrame的索引。由于之前的操作可能导致索引顺序混乱，
# 这一步确保train的索引是从0开始的连续整数。drop=True参数避免将旧索引添加为列。
# inplace=True参数表示在原地修改DataFrame，而不是创建一个新的DataFrame。
train.reset_index(drop=True, inplace=True)


数据准备
# 将test和train两个DataFrame进行拷贝，并通过pd.concat函数合并成一个新的DataFrame，名为data_df。
# 这样做是为了统一处理测试集和训练集的数据。
data_df = pd.concat([test.copy(), train.copy()])

# 在合并的data_df DataFrame中，去除"text"列中的重复行。
# 这一步是为了确保模型训练或测试时使用的文本数据是唯一的，避免重复数据可能带来的影响。
data_df = data_df.drop_duplicates(subset=['text'])

# 重置data_df的索引，确保索引是连续的整数，从0开始。
# 这通常是在对DataFrame进行行的删除或重排操作后的一个常规步骤，以保持索引的一致性和清晰度。
data_df = data_df.reset_index(drop=True)

配置类定义
class cfg:
    LOWERCASE = False  # 定义一个配置项LOWERCASE，指定是否将文本转换为小写，这里设置为False。
VOCAB_SIZE = 300000  # 定义词汇表大小的配置项VOCAB_SIZE，这里设置为300000。这个数值根据具体任务需求设定。

迭代器函数
def train_iter(dataset):
    # 定义一个生成器函数train_iter，接收数据集作为参数，以批量方式迭代数据集中的文本。
    for i in range(0, len(dataset), 1000):
        # 每次迭代返回一个包含1000条文本的批次，直到数据集结束。
        yield dataset[i:i+1000]['text']

特殊令牌定义
# 定义一个特殊令牌列表，包括未知词标记、填充标记、分类开始标记、分隔标记和掩码标记。
# 这些特殊令牌在许多NLP任务中有特定的用途，例如在训练BERT模型时使用。
special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]

数据集转换
# 使用datasets库的Dataset.from_pandas方法，将包含"text"列的data_df DataFrame转换为一个Dataset对象。
# 这个对象更适合于NLP任务中的数据处理和迭代。
dataset = Dataset.from_pandas(data_df[['text']])

创建原始分词器
# 创建一个基于Byte Pair Encoding (BPE) 模型的分词器实例。
# 设置未知词标记为"[UNK]"。
raw_tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))

设置文本标准化
# 设置分词器的文本标准化步骤。
# 使用NFC标准化和（如果cfg.LOWERCASE为True）小写化。
raw_tokenizer.normalizer = normalizers.Sequence(
    [normalizers.NFC()] + [normalizers.Lowercase()] if cfg.LOWERCASE else []
)

设置预分词器
# 设置预分词器为ByteLevel，这在处理基于字节的BPE分词时很常见。
raw_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()

配置和启动训练
# 初始化BPE训练器，并设置词汇表大小和特殊令牌。
trainer = trainers.BpeTrainer(vocab_size=cfg.VOCAB_SIZE, special_tokens=special_tokens)

# 使用train_iter函数迭代提供的数据集进行分词器的训练。
raw_tokenizer.train_from_iterator(train_iter(dataset), trainer=trainer)

转换为预训练分词器
# 创建一个PreTrainedTokenizerFast对象，以便与transformers库兼容。
# 传递了之前训练的分词器对象和特殊令牌的设置。
tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=raw_tokenizer,
    unk_token="[UNK]",
    pad_token="[PAD]",
    cls_token="[CLS]",
    sep_token="[SEP]",
    mask_token="[MASK]",
)

分词测试集
tokenized_test = []  # 初始化一个空列表，用于存储测试集的分词结果。

# 使用tqdm库创建一个进度条，遍历测试集中的文本。tqdm显示进度条，提升用户体验。
for text in tqdm(test['text'].tolist()):
    # 调用tokenizer的tokenize方法对每个文本进行分词，分词结果追加到tokenized_test列表。
tokenized_test.append(tokenizer.tokenize(text))

分词训练集
tokenized_train = []  # 初始化一个空列表，用于存储训练集的分词结果。

# 同样使用tqdm库遍历训练集中的文本，并进行分词，分词结果追加到tokenized_train列表。
for text in tqdm(train['text'].tolist()):
tokenized_train.append(tokenizer.tokenize(text))

定义空操作函数
def dummy(text):
return text

创建并训练TF-IDF向量化器（测试数据）
vectorizer = TfidfVectorizer(
    ngram_range=(3,5),  # 设置n-gram范围为3到5。
    lowercase=cfg.LOWERCASE,  # 根据cfg.LOWERCASE决定是否转换为小写。
    sublinear_tf=True,  # 使用1 + log(tf)缩放频率。
    analyzer='word',  # 设置分析器类型为'word'。
    tokenizer=dummy,  # 使用dummy函数作为分词器。
    preprocessor=dummy,  # 使用dummy函数作为预处理器。
    token_pattern=None,  # 因为使用自定义分词器，所以不使用正则表达式匹配词汇。
    strip_accents='unicode'  # 移除文本中的变音符号。
)
vectorizer.fit(tokenized_test)  # 使用分词后的测试集数据来训练向量化器。
vocab = vectorizer.vocabulary_  # 获取训练后的词汇表。

使用测试集词汇表训练训练集向量化器
vectorizer = TfidfVectorizer(
    ngram_range=(3,5),
    lowercase=cfg.LOWERCASE,
    sublinear_tf=True,
    vocabulary=vocab,  # 使用测试集词汇表。
    analyzer='word',
    tokenizer=dummy,
    preprocessor=dummy,
    token_pattern=None,
    strip_accents='unicode'
)
tf_train = vectorizer.fit_transform(tokenized_train)  # 使用分词后的训练集数据来训练并转换。
tf_test = vectorizer.transform(tokenized_test)  # 转换分词后的测试集数据。

第一步，TfidfVectorizer使用测试集数据来学习词汇表，这有助于了解在评估或测试时认为重要的词汇。
第二步，通过指定vocabulary=vocab参数，确保训练集和测试集使用相同的词汇表进行转换。这样，模型在训练和测试时关注相同的特征空间。
使用自定义的tokenizer和preprocessor（这里是dummy函数），因为数据已经预先分词，不需要TfidfVectorizer进行额外的文本处理。
token_pattern=None是必要的，因为默认的token_pattern只有在不提供tokenizer时才会被使用。设置它为None可以防止TfidfVectorizer尝试使用默认的分词方式。
通过这种方式处理文本数据，可以确保模型训练和测试使用的特征是一致的，同时考虑到数据的实际预处理步骤。

use_GPC = False  # 定义一个布尔变量use_GPC，用于控制是否使用高斯过程分类器进行训练。

# 检查是否需要进行GPC训练。如果use_GPC为False或测试集的样本数量不多于5，则不进行任何操作。
if not use_GPC or len(test.text.values) <= 5:
    pass

else:
    # 如果需要进行GPC训练，首先初始化一个高斯过程分类器对象。
    GPC = GaussianProcessClassifier()
    batch = 2000  # 设置批处理大小为2000，这是在处理数据时每个批次的样本数。

    # 创建一个管道，该管道首先使用TfidfVectorizer向量化文本数据，然后使用TruncatedSVD进行降维。
    svd_vectorizer = Pipeline(steps=[
        ("TfidfVectorizer", vectorizer),  # 向量化步骤。
        ("TruncatedSVD", TruncatedSVD(n_components=100, n_iter=7, random_state=42))  # 降维步骤。
    ])
    
    # 初始化StratifiedKFold，用于之后的交叉验证。这里设置为5折交叉验证，不进行数据洗牌。
    skf = StratifiedKFold(
        n_splits = 5,
        shuffle = False
    )

    svd_list_tr = []  # 初始化一个列表，用于存储训练数据的SVD转换结果。
    svd_list_te = []  # 初始化一个列表，用于存储测试数据的SVD转换结果。

    # 使用tqdm显示进度条，遍历训练数据集，每次处理指定的批量样本。
    for bs in tqdm(range(0, len(train), batch)):
        # 对当前批次的训练数据进行向量化和SVD降维处理，然后将结果添加到svd_list_tr列表。
        svd_train_bs = svd_vectorizer.fit_transform(tokenized_train[bs:bs+batch])
        svd_list_tr.append(svd_train_bs)
        
    # 使用np.concatenate将所有批次的降维结果合并成一个大的数组。
    svd_train = np.concatenate([svd_list_tr[i] for i in range(len(svd_list_tr))])
    goof_train = np.zeros(svd_train.shape[0])  # 初始化一个和svd_train形状相同的全零数组，用于存储训练集的预测结果或其他信息。
    
    # 删除不再需要的变量，以释放内存。
    del svd_train_bs, svd_list_tr
    gc.collect()  # 调用垃圾收集器，尝试回收无用的内存。

# 使用tqdm显示进度条，遍历测试数据集，每次处理一个批次的数据。
for bs in tqdm(range(0, len(test), batch)):
    # 使用之前定义的SVD向量化管道转换当前批次的测试数据，并将结果添加到svd_list_te列表中。
    svd_test_bs = svd_vectorizer.transform(tokenized_test[bs:bs+batch])
    svd_list_te.append(svd_test_bs)

# 使用np.concatenate将所有批次的测试数据转换结果合并成一个大数组。
svd_test = np.concatenate([svd_list_te[i] for i in range(len(svd_list_te))])
# 初始化一个形状与svd_test相同的全零数组，用于存储测试集的预测结果或其他信息。
goof_test = np.zeros(svd_test.shape[0])
# 初始化一个形状为(5, svd_test的样本数)的全零数组，用于存储交叉验证中每一折的测试集预测结果。
goof_test_skf = np.zeros((5, svd_test.shape[0]))

# 删除不再需要的变量以释放内存。
del svd_test_bs, svd_list_te
gc.collect()  # 调用垃圾收集器回收无用内存。

# 从训练数据中提取标签。
y = train['label'].values

# 使用StratifiedKFold进行分层抽样，对数据集进行交叉验证的分割。
for i, (tr_idx, val_idx) in enumerate(skf.split(svd_train, y)):
    # 根据交叉验证的索引，获取当前折的训练数据和验证数据。
    X_train, X_valid = svd_train[tr_idx], svd_train[val_idx]
    y_train, y_valid = y[tr_idx], y[val_idx]
    
    # 深拷贝GPC对象，为当前折的训练准备一个新的分类器。
    gpc_cv = copy.deepcopy(GPC)
    
    # 再次使用tqdm显示进度条，遍历当前折的训练数据，每次处理一个批次的数据进行GPC的拟合。
    for bs in tqdm(range(0, len(X_train), batch)):
        gpc_cv.fit(X_train[bs:batch+bs], y_train[bs:batch+bs])
    
    # 使用拟合好的GPC对验证数据进行预测，并存储预测的概率。
    goof_train[val_idx] = gpc_cv.predict_proba(X_valid)[:,1]
    # 使用拟合好的GPC对整个测试集进行预测，并存储每一折的预测概率。
    goof_test_skf[i, :] = gpc_cv.predict_proba(svd_test)[:,1]

# 计算交叉验证中所有折的测试集预测概率的平均值，并存储为最终的测试集预测结果。
goof_test[:] = goof_test_skf.mean(axis=0)

# 删除不再需要的对象以释放内存。
del svd_vectorizer, svd_train, svd_test, GPC
gc.collect()  # 再次调用垃圾收集器回收无用内存。

# 指定预训练模型的检查点路径。这个路径包含了微调后的模型和配置文件。
model_checkpoint = "/kaggle/input/detect-llm-models/distilroberta-finetuned_v5/checkpoint-13542"

# 从预训练模型检查点加载分词器。分词器用于文本的预处理，将文本转换成模型能理解的格式。
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# 定义一个预处理函数，它将使用上面加载的分词器对输入的文本进行处理。
# 这包括将文本截断或填充至最大长度512，确保所有文本输入具有相同的长度。
def preprocess_function(examples):
    return tokenizer(examples['text'], max_length=512, padding=True, truncation=True)

# 指定模型的标签数量。在这个场景中，因为是二分类任务，所以标签数量为2。
num_labels = 2

# 从预训练模型检查点加载序列分类模型，并指定标签的数量。
# 这个模型将用于文本分类任务，比如判断文本的情感是正面还是负面。
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)

# 检测当前环境是否有可用的GPU。如果有GPU可用，就使用GPU；否则，使用CPU。
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 将模型移动到已确定的设备上（GPU或CPU），以利用GPU加速训练和推理过程。
model.to(device)

# 初始化一个Trainer对象，它将用于管理模型的训练和评估。
# 这里将模型和分词器传递给Trainer，以便在训练过程中使用。
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
)

训练集预处理和预测
# 将训练数据集转换为`datasets`库的`Dataset`对象，这有利于数据的批处理和转换。
train_ds = Dataset.from_pandas(train[['text']])

# 使用之前定义的预处理函数`preprocess_function`对训练数据集进行批处理转换，以适配模型输入要求。
train_ds_enc = train_ds.map(preprocess_function, batched=True)

# 使用`Trainer`对象对编码后的训练数据集进行预测，获取模型输出的原始logits。
train_preds = trainer.predict(train_ds_enc)
logits_tr = train_preds.predictions

# 将原始logits转换为概率。这里，对logits应用softmax函数，计算每个样本属于第一个类的概率。
probs_tr = (np.exp(logits_tr) / np.sum(np.exp(logits_tr), axis=-1, keepdims=True))[:,0]

测试集预处理和预测
# 将测试数据集转换为`Dataset`对象。
test_ds = Dataset.from_pandas(test)

# 使用预处理函数对测试数据集进行批处理转换。
test_ds_enc = test_ds.map(preprocess_function, batched=True)

# 使用`Trainer`对象对编码后的测试数据集进行预测，获取模型输出的原始logits。
test_preds = trainer.predict(test_ds_enc)
logits = test_preds.predictions

# 将原始logits转换为概率。与训练集相同，计算每个样本属于第一个类的概率。
probs = (np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True))[:,0]

这个过程涉及两个主要步骤：数据预处理和模型预测。通过预处理，文本数据被转换成模型能够处理的格式。随后，使用预训练模型对这些数据进行预测，以获取模型对于每个样本的输出（logits）。最后，这些logits被转换成概率值，这些概率值可以直接用于分类任务，或者作为其他机器学习模型输入的特征，进行进一步的分析或堆叠模型训练。这种方法利用了深度学习模型在特征提取方面的强大能力，有助于提高最终模型的性能。

贝叶斯模型 – MultinomialNB
bayes_model = MultinomialNB(alpha=0.02) 
MultinomialNB是多项式朴素贝叶斯分类器，适用于离散特征（比如文本分类中的词频）。
alpha=0.02是平滑参数，用于控制模型复杂度，防止过拟合，通过添加到每个类别的样本数中以避免零概率问题。
随机梯度下降模型 - SGDClassifier
SGD_model = SGDClassifier( max_iter=8000, tol=1e-4, loss="modified_huber" ) 
SGDClassifier用于大规模和稀疏的机器学习问题，通过随机梯度下降（SGD）进行优化。
max_iter=8000设置最大迭代次数。
tol=1e-4是停止准则，如果连续几次迭代改进小于这个值，算法将停止。
loss="modified_huber"指定损失函数，"modified_huber"是一种平滑的hinge损失，适合不平衡分类问题。
k-最近邻模型 - KNeighborsClassifier
kNN_model = KNeighborsClassifier( n_neighbors=10, metric='cosine' ) 
KNeighborsClassifier是基于邻近点的简单分类器，对新的数据点进行分类基于其最接近的k个邻居的类别。
n_neighbors=10指定了邻居的数量。
metric='cosine'使用余弦相似度作为距离度量，适用于文本数据和高维空间。
LightGBM模型 - LGBMClassifier
p6 = {'n_iter': 2500,'verbose': -1,'objective': 'cross_entropy','metric': 'auc', 'learning_rate': 0.00581909898961407, 'colsample_bytree': 0.78, 'colsample_bynode': 0.8, 'lambda_l1': 4.562963348932286, 'lambda_l2': 2.97485, 'min_data_in_leaf': 115, 'max_depth': 23, 'max_bin': 898} LGBM_model = LGBMClassifier(**p6) 
LGBMClassifier是基于梯度提升框架LightGBM的分类器，适用于处理大规模数据，提供了高效的训练速度和较低的内存消耗。
p6字典包含了一系列用于微调模型的参数，如学习率、列的采样比例、正则化项、叶子节点的最小数据量、树的最大深度和最大bin数等，这些参数旨在通过调整以优化模型性能。
CatBoost模型 - CatBoostClassifier
Cat_model = CatBoostClassifier(iterations=2000, verbose=0, l2_leaf_reg=6.6591278779517808, learning_rate=0.005599066836106983, subsample=0.4, allow_const_label=True, loss_function='CrossEntropy') 
iterations=2000: 训练的最大树的数量。
verbose=0: 不打印训练过程中的详细信息。
l2_leaf_reg=6.6591278779517808: L2正则化项的系数。
learning_rate=0.005599066836106983: 学习率。
subsample=0.4: 用于训练每棵树的样本的比例。
allow_const_label=True: 允许所有目标值相同。
loss_function='CrossEntropy': 使用交叉熵作为损失函数。
ExtraTrees模型 - ExtraTreesClassifier
ETR_model = ExtraTreesClassifier( n_estimators=100, criterion='gini' ) 
n_estimators=100: 构建的树的数量。
criterion='gini': 分裂节点时评价分裂质量的指标。
RandomForest模型 - RandomForestClassifier
RF_model = RandomForestClassifier(criterion='entropy') 
criterion='entropy': 使用信息熵作为节点分裂的准则。
高斯过程分类器 - GaussianProcessClassifier
GPC_model = GaussianProcessClassifier() 
默认初始化，适用于较小规模的数据集，因为其计算复杂度较高。
逻辑回归模型 - LogisticRegression
LR_model = LogisticRegression( penalty="elasticnet", solver="saga", max_iter=500, l1_ratio=0.5 ) 
penalty="elasticnet": 使用弹性网正则化。
solver="saga": 优化问题的求解算法。
max_iter=500: 最大迭代次数。
l1_ratio=0.5: L1正则化和L2正则化的混合比例。
XGBoost模型 - XGBClassifier
XGB_model = XGBClassifier( objective='binary:logistic', eval_metric='auc', eta=0.01, ) 
objective='binary:logistic': 二分类的逻辑回归问题，输出概率。
eval_metric='auc': 评价指标为AUC。
eta=0.01: 学习率。

# 检查测试数据集的大小，如果小于等于5，则直接将submission DataFrame输出为CSV文件，不进行后续的堆叠模型训练和预测。
if len(test.text.values) <= 5:
submission.to_csv("submission.csv", index=False)

如果测试集包含超过5个样本，那么执行下面的步骤：
else: # 定义一个预测器列表，这些预测器将用于后续的模型训练和预测。 estimators = [ SGD_model, # SGD分类器 bayes_model, # 多项式朴素贝叶斯分类器 LGBM_model, # LightGBM模型 #Cat_model, # CatBoost模型（这里被注释掉，表示不在当前堆叠中使用） #ETR_model, # Extra Trees模型（同样被注释掉） RF_model, # 随机森林模型 ] # 初始化StratifiedKFold，这是一个交叉验证的方法，用于生成训练/验证数据的索引，保持每个折中类别的比例。 skf = StratifiedKFold( n_splits=5, # 分为5个折 shuffle=False # 不进行洗牌 ) # 从训练数据中提取标签 y = train['label'].values # 初始化两个列表，用于存储训练集和测试集上的模型预测结果，这些结果将用于堆叠模型的训练。 fin_train = [] fin_test = [] 
此代码段是模型堆叠过程的准备阶段，其中estimators列表包含了将要用于生成堆叠特征的基础模型。通过交叉验证（使用StratifiedKFold），这些基础模型的预测结果将用作堆叠模型的输入。此方法是一种常见的集成学习策略，旨在通过组合多个模型的预测结果来提高整体预测性能。

对每个基础模型进行迭代
for est in estimators: 
这一行开始遍历之前定义的estimators列表，每个est代表一个不同的机器学习模型。
初始化Out-of-Fold预测数组
oof_train = np.zeros(tf_train.shape[0]) oof_test = np.zeros(tf_test.shape[0]) oof_test_skf = np.zeros((5, tf_test.shape[0])) 
oof_train用于存储训练集的Out-of-Fold预测结果。
oof_test和oof_test_skf用于存储对测试集的预测结果。oof_test_skf临时存储每一折（fold）的预测结果，之后将对其进行平均以得到最终的测试集预测结果。
使用StratifiedKFold进行交叉验证
for i, (tr_idx, val_idx) in enumerate(skf.split(tf_train, y)): print(f'[CV : {est}] {i+1}/{5}') X_train, X_valid = tf_train[tr_idx], tf_train[val_idx] y_train, y_valid = y[tr_idx], y[val_idx] 
这段代码使用StratifiedKFold对数据进行分层抽样，生成训练集和验证集的索引。这种方法确保了每一折中各类别样本的比例与整个数据集中的比例相同。
拟合模型并进行预测
est_cv = copy.deepcopy(est) est_cv.fit(X_train, y_train) oof_train[val_idx] = est_cv.predict_proba(X_valid)[:,1] oof_test_skf[i, :] = est_cv.predict_proba(tf_test)[:,1] 
对每个基础模型进行深拷贝，以避免在交叉验证的不同迭代中相互影响。
使用训练数据拟合模型，并对验证集和整个测试集进行概率预测。
predict_proba函数的输出是每个样本属于各个类别的概率，这里假设我们关注的是正类（标签为1）的概率。
聚合测试集的预测结果
oof_test[:] = oof_test_skf.mean(axis=0) fin_train.append(oof_train) fin_test.append(oof_test) 
计算测试集上所有折的预测结果的平均值，得到最终的测试集预测概率。
将训练集和测试集的预测结果分别追加到fin_train和fin_test列表中，这些列表最终将被用作堆叠模型的输入特征。
这个过程允许模型从其他模型的预测中学习，通过组合多个模型的强点来提高整体性能，是一种有效的集成学习策略。

整合特征
if use_GPC: fin_train.append(goof_train) fin_test.append(goof_test) fin_train.append(probs_tr) fin_test.append(probs) final_train = np.stack([fin_train[i] for i in range(len(fin_train))], axis=1) final_test = np.stack([fin_test[i] for i in range(len(fin_test))], axis=1) 
如果使用高斯过程分类器（GPC），则将其预测结果加入到最终的训练和测试特征集中。
probs_tr和probs是之前由预训练语言模型得到的预测概率，也被加入作为特征。
使用np.stack将列表中的预测结果沿着新的轴堆叠起来，形成最终的训练集和测试集特征。
训练堆叠模型
Stack_model = VotingClassifier( estimators=[ ("lgb", LGBM_model), ("lr", LR_model), ], weights=[0.65, 0.35], voting='soft', n_jobs=-1 ) Stack_model.fit(final_train, train['label'].values) 
选择了一个VotingClassifier作为堆叠模型，它结合了LightGBM和逻辑回归模型，通过软投票（概率加权平均）来做出最终预测。
weights参数指定了每个模型在最终预测中的权重。
fit方法用于训练堆叠模型，使用之前准备好的特征和训练集的真实标签。
进行预测并生成提交文件
final_preds = Stack_model.predict_proba(final_test)[:,1] submission['generated'] = final_preds.astype(np.float16) submission.to_csv("submission.csv", index=False) 
使用训练好的堆叠模型对测试集进行预测，得到属于正类的概率。
将预测结果存入submission DataFrame中，然后输出为CSV文件，准备提交。