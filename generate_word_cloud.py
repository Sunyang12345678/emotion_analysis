import jieba
import wordcloud
import pandas as pd
class WordCloudObj:
    def __init__(self):
        pass

    def get_cut_movie_review(self):
        """
        得到分词的影评
        :return:
        """
        df = pd.read_csv('./流浪地球/流浪地球.csv', encoding='utf_8_sig', engine='python')
        with open("./流浪地球/stoplist.txt", 'r', encoding='utf-8') as f:
            stoplist = f.read().splitlines()

        # 分词
        data_cut = df['content'].apply(jieba.lcut)

        # 去除停用词
        # 因为之前我们已经将停用词库做成了一个列表，这里只需要去除停用词即可取得需要的影评关键词
        data_new = data_cut.apply(lambda x: [i for i in x if i not in stoplist])
        df_new = pd.DataFrame({"content_cut": data_new})
        df_1 = pd.concat([df, df_new], axis=1)
        return df_1

    def get_grade_movie_review(self, df, keywords):
        """
        得到分级后的影评
        :return:
        """
        text_list = []
        for keyword in keywords:
            content_cut = df[df["evaluate"] == keyword]['content_cut']
            for i in content_cut.tolist():
                text = ','.join(i)
                text_list.append(text)
        return ','.join(text_list)

    def generate_word_cloud(self, movie_reviews, file_path):
        """
        生成词云
        :return:
        """
        w = wordcloud.WordCloud(width=1000, height=700, background_color='white', font_path='msyh.ttc')
        w.generate(movie_reviews)
        w.to_file(file_path)

if __name__ == "__main__":
    good_keywords = ['推荐', '力荐']
    neutral_keywords = ['还行']
    bad_keywords = ['很差', '较差']
    word_cloud_obj = WordCloudObj()
    df = word_cloud_obj.get_cut_movie_review()
    good_movie_reviews = word_cloud_obj.get_grade_movie_review(df, good_keywords)
    word_cloud_obj.generate_word_cloud(good_movie_reviews, "./word_cloud/good_word_cloud.png")
    neutral_movie_reviews = word_cloud_obj.get_grade_movie_review(df, neutral_keywords)
    word_cloud_obj.generate_word_cloud(neutral_movie_reviews, "./word_cloud/neutral_word_cloud.png")
    bad_movie_reviews = word_cloud_obj.get_grade_movie_review(df, bad_keywords)
    word_cloud_obj.generate_word_cloud(bad_movie_reviews, "./word_cloud/bad_word_cloud.png")
