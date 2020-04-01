import gensim
import numpy as np
from numpy.linalg import norm
import jieba
import synonyms
#下载训练后的模型
#model=gensim.models.Word2Vec.load('word2vec//word2vec//word2vec_from_weixin//word2vec//word2vec_wx')
# # fenci//test.word2vec
# # word2vec//word2vec//word2vec_from_weixin//word2vec//word2vec_wx
# file = open('initial_data.txt',"r",encoding="utf8")
# sss=[]
# while True:
#     ss=file.readline().replace('\n','').rstrip()
#     if ss=='':
#         break
#     s1=ss.split(" ")
#     print(s1)
#     sss.append(s1)
#
# file.close()
#
#
# model.build_vocab(sss,update=True)
#model.train(sss,total_examples= model.corpus_count,epochs= model.iter)
# model.save('result_model//test.word2vec')


#model=gensim.models.Word2Vec.load('fenci//test.word2vec')
# f=open("fenci/vocab.txt",mode="a+",encoding="utf-8")
# for word in model.wv.vocab:
#     f.write(word+"\n")



#print(model.similarity("犹豫不决","踌躇"))



# sim_words = model.most_similar(positive=['语量'],topn=10)
# for word,similarity in sim_words:
#     print(word,similarity)

def vector_similarity(s1, s2):
    def sentence_vector(s):
        words = jieba.lcut(s)
        v = np.zeros(256)
        for word in words:
            v += model[word]
        v /= len(words)
        return v

    v1, v2 = sentence_vector(s1), sentence_vector(s2)
    return np.dot(v1, v2) / (norm(v1) * norm(v2))

print(synonyms.compare("食欲","食欲，上升"))
print(synonyms.compare("食欲，下降","食欲，上升"))
print(synonyms.compare("食欲，增加","食欲，上升"))
#print(vector_similarity("心情不错","心情还好"))