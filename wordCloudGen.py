# coding:utf-8

import os
import glob
import numpy as np
from LAC import LAC
from PIL import Image
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS


class WordCloudGen:
    def __init__(self):
        """设置文件路径，设置分词模型"""
        self.txt_folder = os.curdir + '/doc/'
        self.result_folder = os.curdir + '/result/' 

        self.stopwords_path = os.curdir + '/stopwords.txt'
        self.font_path = os.curdir + '/font/msyh.ttf' 
        self.bk_img_path = os.curdir + '/img/RC.jpeg'
        self.bk_img = np.array(Image.open(self.bk_img_path))
        
        self.stop_words = set(STOPWORDS)
        
        self.seg_model = LAC(mode='seg')    # 当下百度的LAC模型分词效果最佳

        self.wc = WordCloud(background_color="white",
            max_words=2000,
            mask=self.bk_img,
            stopwords=self.stop_words,
            font_path=self.font_path,)

    def word_seg(self):
        """完成词频统计与词云序列构建"""
        words_list, words_str, stopwords_set = [], '', set()

        with open(self.stopwords_path, 'r') as sw_f:
            for stopword in sw_f.readlines():
                stopwords_set.add(stopword.strip())

        for file in glob.glob(self.txt_folder+'*.txt'):
            with open(file, 'r') as f:
                seg_result = self.seg_model.run(f.read())
                seg_result = [i for i in seg_result if i not in stopwords_set]
                words_list.extend(seg_result)
                words_str = words_str + ' '.join(seg_result) + ' '

        with open(self.result_folder+'词频.txt', 'w') as wr_f:
            for k,v in Counter(words_list).items():
                wr_f.write("%s,%d\n" % (k,v))

        return words_str

    def plot_word_cloud(self, text):
        """词云绘制"""
        self.wc.generate(text)
        self.wc.to_file(self.result_folder+'result.png')

        plt.imshow(self.wc, interpolation='bilinear')
        plt.axis("off")
        plt.show()


if __name__=='__main__':
    wcg = WordCloudGen()
    words_str = wcg.word_seg()
    wcg.plot_word_cloud(words_str)
    