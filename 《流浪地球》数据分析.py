#!/usr/bin/env python
# coding: utf-8

# ## 数据预处理

# 1、读取停用词库，并进行切片、返回列表  
# 2、对评论数据进行分词  
# 3、去除停用词  

# In[10]:


#导入相关库
import jieba
import numpy as np
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd

# 读取文件信息
data = pd.read_csv('C:/Users/chenqiuhui/Desktop/流浪地球/流浪地球.csv',
                encoding='utf_8_sig',engine='python')

#读入停用词库并进行切片和列表转换
with open(r"C:\Users\chenqiuhui\Desktop\流浪地球\stoplist.txt",'r',encoding='utf-8') as f:
    stoplist = f.read()
stoplist = stoplist.split() + ['\n','',' ']

#分词
data_cut = data['content'].apply(jieba.lcut)

#去除停用词
#因为之前我们已经将停用词库做成了一个列表，这里只需要去除停用词即可取得需要的影评关键词
data_new = data_cut.apply(lambda x:[i for i in x if i not in stoplist])


# In[5]:


#查看数据处理后的新的评论数据
data_new.head(10)


# ### 词频统计

# In[23]:


# 将所有的分词合并
words = []

for content in data_new:
    words.extend(content)

# 创建分词数据框
corpus = pd.DataFrame(words, columns=['word'])
corpus['cnt'] = 1

# 分组统计
g = corpus.groupby(['word']).agg({'cnt': 'count'}).sort_values('cnt', ascending=False)

g.head(10)


# ### 词云图

# In[44]:


# 创建透视表
evaluate = pd.pivot_table(data[['scores','content']],index='scores',aggfunc = np.sum)


# 绘制词云图
good = ['推荐','力荐']
not_bad = ['还行']
terrible = ['很差','较差']
def word_cloud(Data=None):
    data_cut = Data.apply(jieba.lcut)  
    data_new = data_cut.apply(lambda x: [i for i in x if i not in stoplist])  
    #total = pd.Series(_flatten(list(data_new))).value_counts()
    # 将所有的分词合并
    words = []

    for content in data_new:
        words.extend(content)

    # 创建分词数据框
    corpus = pd.DataFrame(words, columns=['word'])
    corpus['cnt'] = 1

    # 分组统计
    g = corpus.groupby(['word']).agg({'cnt': 'count'}).sort_values('cnt', ascending=False)

    g.head(15)
    total = g
    plt.figure(figsize=(10,10))
    mask = plt.imread('rec.jpg')
    wc = WordCloud(font_path='C:/Windows/Fonts/simkai.ttf',mask=mask,background_color='white')
    wc.fit_words(total)
    plt.imshow(wc)
plt.axis('off')


word_cloud(Data=good['content'])    # 好评词云
word_cloud(Data=not_bad['content']   # 中评词云
word_cloud(Data=terrible['content'])   # 差评词云


# ### 用户发表短评数量随日期的变化情况

# In[14]:


# 用户发表短评数量随日期的变化情况
dates = data['times'].apply(lambda x: x.split(' ')[0])
dates = dates.value_counts().sort_index()
plt.figure(figsize=(10,4))
plt.plot(range(len(dates)), dates)
plt.xticks(range(len(dates)), dates.index,rotation=60)
plt.title('用户发表短评数量随日期的变化情况')
plt.ylabel("用户发表短评数量")
plt.show()


# ### 用户发表短评数量随时刻的变化情况

# In[16]:


# 用户发表短评数量随时刻的变化情况
time = pd.to_datetime(data['times']).apply(lambda x: x.hour)
time = time.value_counts().sort_index()
plt.figure(figsize=(10,4))
plt.plot(range(len(time)), time)
plt.xticks(range(len(time)), time.index)
plt.title('用户发表短评数量随时刻的变化情况')
plt.ylabel("用户发表短评数量")
plt.xlabel("时刻")
plt.show()


# ### 评分随日期变化的情况

# In[21]:


# 评分随日期变化的情况
data['times'] = data['times'].apply(lambda x: x.split(' ')[0])
new = pd.DataFrame(0,index=data['times'].drop_duplicates().sort_values(),columns=data['scores'].drop_duplicates().sort_values())
for i, j in zip(data['times'], data['scores']):
    new.loc[i, j] += 1
new = new.iloc[:, :-1]
#可视化
plt.figure(figsize=(10,4))
plt.plot(new)
plt.title('评分随日期变化的情况')
plt.xticks(rotation=90)
plt.legend(["10.0","20.0","30.0","40.0","50.0"],loc="best")
plt.show()


# In[18]:


backgroud_Image = "C:/Users/chenqiuhui/Desktop/中国地图.jpg"
cloud = WordCloud(width=1024, height=768, background_color=None,
         mask=backgroud_Image, font_path="C:/Windows/Fonts/simhei.ttf",
         stopwords=stoplist, max_font_size=400,random_state=50)

plt.imshow(cloud)
plt.axis("off")
plt.show()


# In[48]:


cloud


# In[ ]:


# 观众地域排行榜单（前20）
def draw_bar(comments):
    print("正在处理观众地域排行榜单......")
    data_top20 = Counter(comments['cityName']).most_common(20)  # 筛选出数据量前二十的城市
    bar = Bar('《流浪地球》观众地域排行榜单', '数据来源：猫眼电影 数据分析：16124278-王浩', **style_others.init_style)  # 初始化柱状图
    attr, value = bar.cast(data_top20)  # 传值
    bar.add('', attr, value, is_visualmap=True, visual_range=[0, 16000], visual_text_color='black', is_more_utils=True,
            is_label_show=True)  # 加入数据与其它参数
    bar.render('./output/观众地域排行榜单-柱状图.html')  # 渲染
    print("观众地域排行榜单已完成!!!")
 
 
# lambda表达式内置函数
# 将startTime_tag之前的数据汇总到startTime_tag
def judgeTime(time, startTime_tag):
    if time < startTime_tag:
        return startTime_tag
    else:
        return time
 
 
# 观众评论数量与日期的关系
def draw_DateBar(comments):
    print("正在处理观众评论数量与日期的关系......")
    time = pd.to_datetime(comments['startTime'])  # 获取评论时间并转换为标准日期格式
    time = time.apply(lambda x: judgeTime(x, startTime_tag))  # 将2019.2.4号之前的数据汇总到2.4 统一标识为电影上映前影评数据
    timeData = []
    for t in time:
        if pd.isnull(t) == False:  # 获取评论日期（删除具体时间）并记录
            t = str(t)  # 转换为字符串以便分割
            date = t.split(' ')[0]
            timeData.append(date)
 
    data = Counter(timeData).most_common()  # 记录相应日期对应的评论数
    data = sorted(data, key=lambda data: data[0])  # 使用lambda表达式对数据按日期进行排序
 
    bar = Bar('《流浪地球》观众评论数量与日期的关系', '数据来源：猫眼电影 数据分析：16124278-王浩', **style_others.init_style)  # 初始化柱状图
    attr, value = bar.cast(data)  # 传值
    bar.add('', attr, value, is_visualmap=True, visual_range=[0, 43000], visual_text_color='black', is_more_utils=True,
            is_label_show=True)  # 加入数据和其它参数
    bar.render('./output/观众评论日期-柱状图.html')  # 渲染
    print("观众评论数量与日期的关系已完成!!!")
 
 
# 观众情感曲线
def draw_sentiment_pic(comments):
    print("正在处理观众情感曲线......")
    score = comments['score'].dropna()  # 获取观众评分
    data = Counter(score).most_common()  # 记录相应评分对应的的评论数
    data = sorted(data, key=lambda data: data[0])  # 使用lambda表达式对数据按评分进行排序
    line = Line('《流浪地球》观众情感曲线', '数据来源：猫眼电影 数据分析：16124278-王浩', **style_others.init_style)  # 初始化
    attr, value = line.cast(data)  # 传值
 
    for i, v in enumerate(attr):  # 将分数修改为整数便于渲染图上的展示
        attr[i] = v * 2
 
    line.add("", attr, value, is_smooth=True, is_more_utils=True, yaxis_max=380000, xaxis_max=10)  # 加入数据和其它参数
    line.render("./output/观众情感分析-曲线图.html")  # 渲染
    print("观众情感曲线已完成!!!")
 
 
# 观众评论数量与时间的关系图
def draw_TimeBar(comments):
    print("正在处理观众评论数量与时间的关系......")
    time = comments['startTime'].dropna()  # 获取评论时间
    timeData = []
    for t in time:
        if pd.isnull(t) == False:  # 获取评论时间（当天小时）并记录
            time = t.split(' ')[1]
            hour = time.split(':')[0]
            timeData.append(int(hour))  # 转化为整数便于排序
 
    data = Counter(timeData).most_common()  # 记录相应时间对应的的评论数
    data = sorted(data, key=lambda data: data[0])  # 使用lambda表达式对数据按时间进行排序
 
    bar = Bar('《流浪地球》观众评论数量与时间的关系', '数据来源：猫眼电影 数据分析：16124278-王浩', **style_others.init_style)  # 初始化柱状图
    attr, value = bar.cast(data)  # 传值
    bar.add('', attr, value, is_visualmap=True, visual_range=[0, 40000], visual_text_color='black', is_more_utils=True,
            is_label_show=True)  # 加入数据和其它参数
    bar.render('./output/观众评论时间-柱状图.html')  # 渲染
    print("观众评论数量与时间的关系已完成!!!")
 
 
# 观众评论走势与时间的关系
def draw_score(comments):
    print("正在处理观众评论走势与时间的关系......")
    page = Page()  # 页面储存器
    score, date, value, score_list = [], [], [], []
    result = {}  # 存储评分结果
 
    d = comments[['score', 'startTime']].dropna()  # 获取评论时间
    d['startTime'] = d['startTime'].apply(lambda x: pd.to_datetime(x.split(' ')[0]))  # 获取评论日期（删除具体时间）并记录
    d['startTime'] = d['startTime'].apply(lambda x: judgeTime(x, startTime_tag))  # 将2019.2.4号之前的数据汇总到2.4 统一标识为电影上映前影评数据
 
    for indexs in d.index:  # 一种遍历df行的方法（下面还有第二种，iterrows）
        score_list.append(tuple(d.loc[indexs].values[:]))  # 评分与日期连接  转换为tuple然后统计相同元素个数
    print("有效评分总数量为：", len(score_list), " 条")
    for i in set(list(score_list)):
        result[i] = score_list.count(i)  # dict类型，统计相同日期相同评分对应数
 
    info = []
    for key in result:
        score = key[0]  # 取分数
        date = key[1]  # 日期
        value = result[key]  # 数量
        info.append([score, date, value])
    info_new = pd.DataFrame(info)  # 将字典转换成为数据框
    info_new.columns = ['score', 'date', 'votes']
    info_new.sort_values('date', inplace=True)  # 按日期升序排列df，便于找最早date和最晚data，方便后面插值


# ### 评论来源城市分析

# In[32]:


import pandas as pd
from pyecharts.charts import Bar
from pyecharts import Style

# 读取合并后的文件信息
data = pd.read_csv('C:/Users/chenqiuhui/Desktop/流浪地球/流浪地球.csv',
                encoding='utf_8_sig',engine='python')
# 查看数据的基本信息属性
data.describe()

# 统计各城市出现的数据信息
ct_data = pd.DataFrame(data['citys'])
# 将城市名称为空的数据删除
#ct_data = ct_datas.dropna(axis=0)
#print(ct_data.size)


# 定义样式
style = Style(
    title_color='#fff',
    title_pos='center',
    width=800,
    height=500,
    background_color='#404a59',
    subtitle_color='#fff'
)

# 根据城市数据生成柱状图
city_counts = ct_data.groupby("citys").size()
city_sorted = city_counts.sort_values(ascending=False).head(20)

bar = Bar("", "", **style.init_style)
attr, value = bar.cast(list(city_sorted.items()))
bar.add("", attr, value, is_visualmap=True, visual_range=[0, 2500],
        visual_text_color='#fff', label_color='#fff',
        xaxis_label_textcolor='#fff', yaxis_label_textcolor='#fff', 
        is_more_utils=True,is_label_show=True)

# 保存可视化图片文件
bar.render("城市排行-柱状图.html")
# 图像可视化
bar


# In[35]:


# pandas读取会员的评分数据
df_score =  pd.DataFrame(data['scores'])
# 将评分为空的数据删除
df_score = df_score.dropna(axis=0)
print(df_score.size)
# 构建Pie图生成函数
def genPiePicture(df_score):
    pass   
    return pie
# 生成男性的评分pie图
pie = genPiePicture(df_score)
pie


# In[ ]:





# ## 情感分类

# In[ ]:





# In[ ]:





# In[ ]:




