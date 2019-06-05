# coding = utf-8 #

# 对文本进行jieba分词
import jieba
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn import svm
from sklearn.metrics import roc_auc_score,precision_score

def drop_stopwords(contents, stopwords):
    contents_clean = []
    for line in contents:
        line_clean = []
        for word in line:
            if word == " ":
                continue
            if word in stopwords:
                continue
            line_clean.append(word)
        contents_clean.append(line_clean)
    return contents_clean

class myDM:
    #标签矩阵 string -- num
    label_dict = {}
    trainData = [] #训练数据集
    trainLabel = [] #训练标签集合
    testData =[] #测试数据集和
    resLabel =[] #最终的返回的结果的集合
    stopWordSet = set()             #读取停用词表，有两个
    tfidf_vec = None             #
    tfidf_matrix = None

    def __init__(self):
        self.label_dict = self.read_Label_Dict("emoji.data")
        self.trainLabel = self.read_Label_Mat("train.solution")
        self.trainData = self.read_Train_Mat("train.data")
        self.stopWordSet = self.read_Stop_word()
        self.processData()

    #读取便签字典
    def read_Label_Dict(self,filename):
        dict = {}
        with open(filename,'r', encoding='utf-8') as f:
            for line in f:
                #进行转化
                line.replace("\ufeff","")
                temp = line.split()
                label_name = temp[1]
                label_number = int(temp[0])
                dict[label_name] = label_number #构建字典集
            f.close()
        return dict

    #读取标签矩阵
    def read_Label_Mat(self,filename):
        res = []
        with open(filename,'r', encoding='utf-8') as f:
            for line in f:
                t = line.split()
                str = t[0].replace("{","")
                str = str.replace("}","")
                res.append(self.label_dict[str])
        return res

    #读取训练数据
    def read_Train_Mat(self,filename):
        res = []
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.replace("\n","")
                res.append(line)
        return res

    def read_Stop_word(self): #读取两个停词表
        res = []
        with open('./stopword/stopwords.txt', 'r', encoding='utf-8') as f:
            for line in f:
                line = line.replace("\n","")
                if line != "":
                    res.append(line)
        with open('./stopword/stopwords2.txt', 'r', encoding='utf-8') as f:
            for line in f:
                line = line.replace("\n","")
                if line != "":
                    res.append(line)
        return set(res)

    #对于读取的训练数据进行预处理
    def processData(self):
        #对于Data矩阵进行预处理,得到实矩阵
        #cut_words = list(map(lambda s: list(jieba.cut(s)), self.trainData))
        #afterRemoveStop = drop_stopwords(cut_words,self.stopWordSet) #去除停词表
        #取前2000个数据
        test_2000_data = []
        test_2000_label = []
        for i in range(10000):
            test_2000_data.append(self.trainData[i])
            test_2000_label.append(self.trainLabel[i])

        cut_words = list(map(lambda s: list(jieba.cut(s)), test_2000_data))
        afterRemoveStop = drop_stopwords(cut_words,self.stopWordSet) #去除停词表
        print(afterRemoveStop)

        t = []
        for i in afterRemoveStop:
            str = ""
            for j in i:
                str = str + " " + j
            t.append(str)
        print(t)

        self.tfidf_vec = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", max_df=0.5)
        self.tfidf_matrix = self.tfidf_vec.fit_transform(t)
        # # 得到语料库所有不重复的词
        print(self.tfidf_vec.get_feature_names())
        # # 得到每个单词对应的id值
        print(self.tfidf_vec.vocabulary_)
        tfidf_my = self.tfidf_matrix.toarray()
        #进行数据降维
        print(tfidf_my.shape)
        pcaClf = PCA(n_components= 2000, whiten=True)
        data_PCA = pcaClf.fit_transform(tfidf_my)
        print(data_PCA.shape)
        #进行预测
        #利用支持向量积
        model = svm.SVC(C=2, kernel='rbf', gamma='auto', decision_function_shape='ovr')
        model.fit(data_PCA, test_2000_label)
        res_label = model.predict(data_PCA)

        print(precision_score(np.array(test_2000_label),res_label, average="micro"))

        #学习函数
    def fit(self):
        pass

    #预测函数
    def predict(self):
        pass

#Main函数
if __name__ == '__main__':
    dm = myDM()

