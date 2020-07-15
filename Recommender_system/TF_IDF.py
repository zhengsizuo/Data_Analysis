import jieba
import math
import jieba.analyse

class TF_IDF:
    def __init__(self, file, stop_file):
        self.file = file
        self.stop_file = stop_file
        self.stop_words = self.getStopWords()
        self.dMap = self.loadData()

    # 获取停用词列表
    def getStopWords(self):
        swlist = list()
        for line in open(self.stop_file, "r", encoding="utf-8").readlines():
            swlist.append(line.strip())
        print("加载停用词完成...")
        return swlist

    # 加载商品和其对应的短标题，使用jieba进行分词并去除停用词
    def loadData(self):
        dMap = dict()
        for line in open(self.file, "r", encoding="utf-8").readlines():
            id, title = line.strip().split("\t")
            dMap.setdefault(id, [])
            for word in list(jieba.cut(str(title).replace(" ", ""), cut_all=False)):
                if word not in self.stop_words:
                    dMap[id].append(word)
        print("加载商品和对应的短标题，并使用jieba分词和去除停用词完成...")
        return dMap

    # 计算单词t在文档D_i中出现的频率, term frequency
    def TF(self, t, D_i):
        return D_i.count(t) / len(D_i)

    # 计算单词t在整个语料库的inverse document frequency
    def IDF(self, t):
        dMap = self.dMap
        N = len(dMap)  # 文档数目
        I = 0
        for id, doc in dMap.items():
            if t in doc:
                I += 1

        return math.log(N / (1+I))

    # 计算编号为id的文档D_i每个单词的TF-IDF值
    def total_TF_IDF(self, id):
        ret_dict = dict()
        dMap = self.dMap
        doc = dMap[id]
        for word in doc:
            tf_idf_v = self.TF(word, doc) * self.IDF(word)
            ret_dict.setdefault(word, tf_idf_v)

        # print(ret_dict)
        return sorted(ret_dict.items(), key=lambda x: x[1], reverse=True)


if __name__ == '__main__':
    phone_file = 'data/phone-title/id_title.txt'
    stop_file = 'data/phone-title/stop_words.txt'
    tf_idf = TF_IDF(phone_file, stop_file)
    for key in tf_idf.dMap:
        print(key, tf_idf.total_TF_IDF(key))
