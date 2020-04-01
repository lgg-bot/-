from extract_answerkey import *
from sim_cilin import *
from lstm_predict import *
from apiTest import *


import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8') #改变标准输出的默认编码
'''问答类'''
class ChatBotGraph:
    def __init__(self):
        self.extract_answerkey = extract_answerkey()
        self.sim_cilin = SimCilin()
        self.lstm_predict=LSTMNER()
        self.apiTest=apiTest()


    def chat_main(self, sent):
        # 获取关键词
        chars,tags=self.lstm_predict.predict(sent)
        key1=self.lstm_predict.match_key(chars,tags)
        key2=self.extract_answerkey.ti_idf(sent)
        key=key1+key2
        if len(key) != 0:
            result_key=set(key)#最终关键词
            tran_key=(",".join(list(result_key)))#关键词拼接去寻找相似症状

            #获取句子正负情绪
            sentiment_result=self.apiTest.sentiment(sent)

            #获取相似症状
            final_scores,res=self.sim_cilin.key_match_finalkey(tran_key)

            #获取程度词，时间词，频率词
            other_result=self.extract_answerkey.match_otheranswer(sent)
            print("结果".center(50, "*"))
            if sentiment_result[2]>0.9:#积极
                print("关键词：",result_key)
                print(("相似的症状的评分: " + str(final_scores)))
                print(("相似的症状: " + str(res)))
                print("程度词，时间词，频率词:",other_result)
            else:
                print(("相似的症状的评分: " + str(final_scores)))
                print(("相似的症状: " + str(res)))
                print("程度词，时间词，频率词:",other_result)
            print("end".center(54, "-"), "\n")
        else:
            print("抱歉，没有匹配到关键词")




if __name__ == '__main__':
    handler = ChatBotGraph()
    while 1:
        question = input('用户:')
        answer = handler.chat_main(question)


