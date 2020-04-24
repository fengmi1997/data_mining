import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

# 数据挖掘大作业
# by 翟艺伟，学号：2018201653


# 读取excel
pf = pd.read_excel('grade_original.xls', sheet_name='Sheet0')
# 删除指定列，包括
pf.drop(['课程序号', '课程代码', '授课教师', '实得学分', '绩点', '课程类别代码', '修读类别', '学生类别'], axis=1, inplace=True)
# 在学期列里面将秋转化为１，春转化为２
pf.ix[pf['学期'] == '秋', '学期'] = 1
pf.ix[pf['学期'] == '春', '学期'] = 2
# 筛选出14级两个班级的数据
data = pf[pf['班级'].isin(['08031401', '08031402', '08031403'])]
# 去掉这些班里面上一级13级的数据
data = data[data['学号'] > 2014300000]
# 将2014级学生成绩数据按照学号从小到达排列
# True  升序排列   False 降序排列
data.sort_values(by=["学号", "学年度", "学期"], inplace=True, ascending=[True, True, True])
# 索引从0编号
data.reset_index(drop=True, inplace=True)
# 14级学生数量
# count_student = dict(data['学号'].value_counts())
# print(count_student)
data.drop(['学期', '班级', '课程名称', '课程类别'], axis=1, inplace=True)
data['课程学分×分数'] = data.apply(lambda x: x['课程学分'] * x['分数'], axis=1)
data = data[~data['学年度'].isin(['2018-2019'])]
data.to_excel('grade.xls')
# 新建一个Data frame，索引值是学生id
data.drop(['学年度', '分数', '课程学分', '课程学分×分数'], axis=1, inplace=True)
# 总共有多少种类别的课程
# count_class = dict(data['课程类别'].value_counts())
# print(count_class)
# 去掉重复行
data.drop_duplicates('学号', keep='first', inplace=True)
data.to_excel('grade_final.xls')

# 计算每个学生每个学年的学分绩
df = pd.read_excel('grade.xls', sheet_name='Sheet1')
id_year_group = (df.groupby(['学号', '学年度']))
id_year_data_frame = id_year_group.sum()
id_year_data_frame['学分绩'] = id_year_data_frame.apply(lambda x: x['课程学分×分数'] / x['课程学分'], axis=1)
id_year_data_frame.drop(['Unnamed: 0', '课程学分', '分数', '课程学分×分数'], axis=1, inplace=True)

id_year_data_frame.to_excel('grade_process.xls')
score = id_year_data_frame['学分绩']
length_score = np.size(score) / 4.0
score_num = np.array(score).reshape(int(length_score), 4)
grade_final = pd.DataFrame(score_num)

# 读入有学号的data frame记为student_frame，与grade_final合并
student_frame = pd.read_excel('grade_final.xls', sheet_name='Sheet1')
student_frame.drop(['Unnamed: 0'], axis=1, inplace=True)
grade_mining = pd.concat([student_frame, grade_final], axis=1)
grade_mining.rename(columns={0: '第一学年学分绩', 1: '第二学年学分绩', 2: '第三学年学分绩', 3: '第四学年学分绩'}, inplace=True)


# 用平均值来代替缺失值
grade_mining = grade_final
grade_mining[grade_mining == 0] = np.NaN
# grade_mining.fillna(method='pad', axis=1, inplace=True)
for column in list(grade_mining.columns[grade_mining.isnull().sum() > 0]):
    mean_val = grade_mining[column].mean()
    grade_mining[column].fillna(mean_val, inplace=True)

grade_mining.rename(columns={0: '第一学年学分绩', 1: '第二学年学分绩', 2: '第三学年学分绩', 3: '第四学年学分绩'}, inplace=True)

# 使用k-means进行聚类分析
k = 4  # 聚类的类别
iteration = 500  # 聚类的最大循环数
grade_mining_zs = 1.0 * (grade_mining - grade_mining.mean()) / grade_mining.std()  # 数据标准化
model = KMeans(n_clusters=k, n_jobs=4, max_iter=iteration)
model.fit(grade_mining_zs)

# 简单打印结果
r1 = pd.Series(model.labels_).value_counts()
r2 = pd.DataFrame(model.cluster_centers_)
r = pd.concat([r2, r1], axis=1)
r.columns = list(grade_mining.columns) + [u'类别数目']
print(r)

# 详细输出原始数据及其类别
r_detail = pd.concat([grade_mining, pd.Series(model.labels_, index=grade_mining.index)], axis=1)
r_detail.columns = list(grade_mining.columns) + [u'聚类类别']
print(r_detail)


# 自定义作图函数
def density_plot(data):
    # 用来正常显示中文标签和负号
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    p = data.plot(kind='kde', linewidth=2, subplots=True, sharex=False)
    [p[i].set_ylabel(u'密度') for i in range(data.shape[1])]
    plt.legend()
    return plt


# 作概率密度图
fig_output = '../tmp/2014'
for i in range(k):
    data_r = grade_mining[r_detail[u'聚类类别'] == i].iloc[:, 0:]
    density_plot(data_r).savefig(u'%s%s' % (fig_output, i))
plt.close()

#  对k means结果可视化展示
tsne = TSNE()
# 数据降维度
tsne.fit_transform(grade_mining_zs)
tsne = pd.DataFrame(tsne.embedding_, index=grade_mining_zs.index)


plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


# 不同类别用不同颜色和样式绘图
plt.figure(figsize=(6, 5))
d = tsne[r_detail[u'聚类类别'] == 0]
plt.plot(d[0], d[1], 'r.')
d = tsne[r_detail[u'聚类类别'] == 1]
plt.plot(d[0], d[1], 'go')
d = tsne[r_detail[u'聚类类别'] == 2]
plt.plot(d[0], d[1], 'b*')
d = tsne[r_detail[u'聚类类别'] == 3]
plt.plot(d[0], d[1], 'y*')
d = tsne[r_detail[u'聚类类别'] == 4]
plt.plot(d[0], d[1], 'mo')
plt.savefig('../tmp/2014', bbox_inches='tight', dpi=300)

