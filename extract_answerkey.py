#-*-coding:utf-8-*-
import os
import csv
import ahocorasick
import jieba.analyse
import gensim
import heapq


class extract_answerkey:
    def __init__(self):
        cur_dir = '/'.join(os.path.abspath(__file__).split('/')[:-1])#os.path.abspath(__file__)返回当前脚本的绝对路径
        self.time_recent_dict=self.create_dict(cur_dir+"词典分类/时间/recent.txt")#时间 近
        self.time_furthest_dict=self.create_dict(cur_dir+"词典分类/时间/furthest.txt")#时间 远
        self.deny_dict=self.create_dict(cur_dir+"词典分类/否定/否定.txt")#否定
        self.degree_deep_dict=self.create_dict(cur_dir+"词典分类/程度/deep.txt")#程度 深
        self.degree_shallow_dict=self.create_dict(cur_dir+"词典分类/程度/shallow.txt")#程度 浅
        self.frequency_high_dict=self.create_dict(cur_dir+"词典分类/频率/fre_high.txt")#频率 高
        self.frequency_low_dict=self.create_dict(cur_dir+"词典分类/频率/fre_low.txt")#频率 低
        self.sum_symptom=self.create_dict(cur_dir+"词典分类/症状/症状总.txt")#所有的症状
        self.sure_symptom=self.create_dict(cur_dir+"词典分类/症状/确定的症状.txt")#后半部分确定的症状
        self.synonymy_symptom=self.create_synonymy_dict()#前半部分同义词
        self.stopwords=self.create_dict(cur_dir+"词典分类/stopwords.txt")#停用词
        self.wdtype_dict=self.build_wdtype_dict()
        self.model = gensim.models.Word2Vec.load('fenci//test.word2vec')
        self.vocab = self.create_dict(cur_dir + "fenci/vocab.txt")  # 词向量模型中所有的词
        # print(self.time_recent_dict)
        # print(self.time_furthest_dict)
        # print(self.deny_dict)
        # print(self.degree_deep_dict)
        # print(self.degree_shallow_dict)
        # print(self.frequency_high_dict)
        # print(self.frequency_low_dict)
        # print(self.sum_symptom)
        # print(self.sure_symptom)
        # print(self.synonymy_symptom)
        # print(self.stopwords)

    '''创建词典'''
    def create_dict(self,path):
        a=[]
        f=open(path,"r",encoding="utf-8")
        lines=f.readlines()
        for line in lines:
            a.append(line.split("\n")[0])
        return (set(a))

    '''创建同义词词典'''
    def create_synonymy_dict(self):
        path = "词典构建//synonymDictionary_症状.csv"
        csvfile = open(path, "r")
        reader = csv.reader(csvfile)
        column1 = [row[1:] for row in reader]
        # print(column1)
        synonym_dict = dict()
        for row in column1:
            for l in row[1:]:
                if l != "":
                    synonym_dict[l] = row[0]
        return synonym_dict

    '''ac自动机加速'''
    def build_actree(self,wordlist):
        actree = ahocorasick.Automaton()
        for index, word in enumerate(wordlist):
            actree.add_word(word, (index, word))
        actree.make_automaton()
        return actree

    '''建立类型字典'''
    def build_wdtype_dict(self):
        wd_dict = dict()
        region_words=self.degree_deep_dict|self.degree_shallow_dict|self.frequency_high_dict|\
                     self.frequency_low_dict|self.time_furthest_dict|self.time_recent_dict#|self.stopwords
        for wd in region_words:
            wd_dict[wd] = []
            if wd in self.degree_deep_dict:
                wd_dict[wd].append('degree_deep')
            if wd in self.degree_shallow_dict:
                wd_dict[wd].append('degree_shallow')
            if wd in self.frequency_high_dict:
                wd_dict[wd].append('frequency_high')
            if wd in self.frequency_low_dict:
                wd_dict[wd].append('frequency_low')
            if wd in self.time_furthest_dict:
                wd_dict[wd].append('time_furthest')
            if wd in self.time_recent_dict:
                wd_dict[wd].append('time_recent')
            # if wd in self.stopwords:
            #     wd_dict[wd].append('stopwords')

        return wd_dict

    '''求一个list中最大5个值及其索引'''

    def max_number(self, m):
        # m = [9, 8, 8, 6, 6, 5, 4, 2, 2, 0]
        max_number = heapq.nlargest(5, m)
        max_index = []
        for t in max_number:
            if t > 0.5:
                index = m.index(t)
                max_index.append(index)
                m[index] = -1
        return max_number, max_index

    '''比较计算词语之间的相似度'''
    def compute_word_sim(self, word1, word2):
        return self.model.similarity(word1, word2)

    '''基于词相似度计算句子相似度'''

    def distance(self, text1, text2):
        filename = "stopwords.txt"
        p = "症状分词.txt"

        #jieba.load_userdict(p)
        #jieba.analyse.set_stop_words(filename)
        twords1 = jieba.analyse.extract_tags(text1, topK=3)
        twords2 = jieba.analyse.extract_tags(text2, topK=3)

        #print(text1, "分词为:", twords1)
        #print(text2, "分词为:", twords2)

        words1 = []
        words2 = []
        for word in twords1:
            if word in self.vocab:
                words1.append(word)
        for word in twords2:
            if word in self.vocab:
                words2.append(word)

        #print(text1, "分词为:", words1)
        #print(text2, "分词为:", words2)
        if len(words1) == 0 or len(words2) == 0:
            return 0

        # words1 = [word.word for word in pseg.cut(text1) if word.flag[0] not in ['u', 'x', 'w']]
        # words2 = [word.word for word in pseg.cut(text2) if word.flag[0] not in ['u', 'x', 'w']]
        # print(text1,"分词为:",words1)
        # print(text2,"分词为:",words2)
        score_words1 = []
        score_words2 = []
        for word1 in words1:
            score = max(self.compute_word_sim(word1, word2) for word2 in words2)
            score_words1.append(score)
        for word2 in words2:
            score = max(self.compute_word_sim(word2, word1) for word1 in words1)
            score_words2.append(score)
        similarity = max(sum(score_words1) / len(words1), sum(score_words2) / len(words2))

        return similarity

    '''匹配关键词'''
    def match_key(self,actree,question):
        region_wds = []
        for i in actree.iter(question):
            wd = i[1][1]
            region_wds.append(wd)
        stop_wds = []
        for wd1 in region_wds:#最长匹配规则
            for wd2 in region_wds:
                if wd1 in wd2 and wd1 != wd2:
                    stop_wds.append(wd1)
        final_wds = [i for i in region_wds if i not in stop_wds]
        final_dict = {i: self.wdtype_dict.get(i) for i in final_wds}
        return final_dict

    '''将list中的词按照text中出现的顺序排序'''
    def sort_sen(self,text,list):#将list中的词按照text中出现的顺序排序
        c = []
        for x in list:
            c.append(text.index(x))
        #print(c)
        d = []
        for x in c:
            d.append(x)
        d.sort()
        #print(d)
        res = []
        for x in d:
            # print(x)
            # print(c.index(x))
            res.append(list[c.index(x)])
        return res

    '''匹配频率，程度，时间词'''
    def match_otheranswer(self,text):

        result=[]
        #匹配时间词
        wordlist=list(self.time_recent_dict | self.time_furthest_dict)
        actree=self.build_actree(wordlist)
        time=self.match_key(actree,text)
        for l in time.keys():
            text=text.replace(l,"")
        #print(time,text)
        result.append(time)

        #匹配频率词
        wordlist = list(self.frequency_low_dict | self.frequency_high_dict)
        actree = self.build_actree(wordlist)
        frequency = self.match_key(actree, text)
        for l in frequency.keys():
            text = text.replace(l, "")
        #print(frequency, text)
        result.append(frequency)

        # 匹配程度词
        wordlist = list(self.degree_shallow_dict | self.degree_deep_dict)
        actree = self.build_actree(wordlist)
        degree = self.match_key(actree, text)
        for l in degree.keys():
            text = text.replace(l, "")
        #print(degree, text)
        result.append(degree)

        return result

        # #去除停用词
        # wordlist = list(self.stopwords)
        # actree = self.build_actree(wordlist)
        # denywords = self.match_key(actree, answer)
        # for l in denywords.keys():
        #     answer = answer.replace(l, "")
        # print(denywords, answer)

        # #提取症状
        # p = "症状分词.txt"
        # jieba.load_userdict(p)
        # filename="词典分类/stopwords.txt"
        # jieba.analyse.set_stop_words(filename)#在提取的时候不会把停用词提取出来
        # Key=jieba.analyse.extract_tags(answer,topK=3)
        # if "哭" in answer:#哭提取不出来，单独处理
        #     Key.append("哭")
        # if "死" in answer:#哭提取不出来，单独处理
        #     Key.append("死")
        # print("原始的key:",Key)
        #
        # symptom_key=[]
        # transfer_symptom_key=[]
        # final_symptom_key=[]
        # for key in Key:
        #     if key in self.sum_symptom:
        #         symptom_key.append(key)
        # symptom_key = self.sort_sen(answer, symptom_key)
        # print("经过筛选后得到的key:",symptom_key)
        #
        # for key in symptom_key:#进行同义词转换
        #     if key in self.synonymy_symptom.keys() and key not in list(self.synonymy_symptom.values()):
        #         transfer_symptom_key.append(self.synonymy_symptom[key])
        #     else:
        #         transfer_symptom_key.append(key)
        # transfer_symptom_key=set(transfer_symptom_key)
        # print("经过转换的key:",transfer_symptom_key)

        #映射到本地词典
        # for k in transfer_symptom_key:
        #     for sym in self.sure_symptom:
        #         if k in sym:
        #             final_symptom_key.append(sym)
        # final_symptom_key=set(final_symptom_key)
        # print("最终的key:",final_symptom_key)
        # sym=""
        # for k in transfer_symptom_key:
        #     sym+=k
        # if sym in self.sure_symptom:#针对心情不好，食欲下降这种的
        #     final_symptom_key.append(sym)
        #     final_symptom_key=set(final_symptom_key)
        #     print("最终的key:",final_symptom_key)
        #     return(final_symptom_key)
        # else:
        #     print(transfer_symptom_key)
        #     if transfer_symptom_key < self.vocab:#代表词全出现在了词向量文本中
        #         transfer_text=(",".join(list(transfer_symptom_key)))#症状转化为逗号连接，方便后面分词
        #         f = open("症状.txt", encoding="utf-8")
        #         b = []
        #         lines = f.readlines()
        #         for line in lines:
        #             b.append(line.split("\n")[0])
        #         f.close()
        #
        #         score = []
        #         for sym in b:
        #             # print(dd)
        #             t = self.distance(sym, transfer_text)
        #             score.append(t)
        #         if 0 in score:  # 代表分词失败或者没有找到相似的
        #             print((transfer_text + " 分词失败" + "\n"))
        #         scores, index = self.max_number(score)
        #         if len(scores) == 0:  # 代表没有找到相似的
        #             # f.write((sym + " 相似的症状的评分: " + str(scores) + "\n"))
        #             print((transfer_text + " 相似的症状:无" + "\n"))
        #         else:
        #
        #             for i in index:
        #                 final_symptom_key.append(b[i])
        #             print((transfer_text + " 相似的症状的评分: " + str(scores)))
        #             print((transfer_text + " 相似的症状: " + str(final_symptom_key)))
        #             return (set(final_symptom_key))
        #     else:#只能采用传统的关键词包含方式了
        #         for k in transfer_symptom_key:
        #             for sym in self.sure_symptom:
        #                 if k in sym:
        #                     final_symptom_key.append(sym)
        #         final_symptom_key=set(final_symptom_key)
        #         print("最终的key:",final_symptom_key)
        #         return(final_symptom_key)

    '''匹配出用户语言的关键词'''
    def ti_idf(self,text):
        p = "症状分词.txt"
        jieba.load_userdict(p)
        #txt = "我经常看到别人哭就会感觉伤心"
        # print("|".join(jieba.cut(txt)))
        # print([word.word for word in pseg.cut(txt)])
        file_name = "stopwords.txt"
        jieba.analyse.set_stop_words(file_name)
        Key = jieba.analyse.extract_tags(text, topK=2)
        print("ti_idf_Key:",Key)
        return Key

    '''将一个list中的涉及到包含的内容中去掉被包含的内容，用在ner+ti_idf以后进行过滤'''
    def longest_match(self,list):
        stop_wds = []
        for wd1 in list:  # 最长匹配规则
            for wd2 in list:
                if wd1 in wd2 and wd1 != wd2:
                    stop_wds.append(wd1)
        final_wds = [i for i in list if i not in stop_wds]
        return final_wds


# if __name__ == '__main__':
#     handler = extract_answerkey()
#     while 1:
#         answer=input("输入:")
#         handler.match_answer(answer)


