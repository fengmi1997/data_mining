import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn import metrics
from sklearn.metrics import silhouette_score

# 数据挖掘大作业
# by 翟艺伟，学号：2018201653

# 读取excel
pf = pd.read_excel('id_curriculum.xls', sheet_name='Sheet1')
# 对初步生成的数据，进行进一步的数据处理，产生下一步挖掘的数据备份
# 删除空值很多的列
pf.loc['Row_sum'] = pf.apply(lambda x: x.sum())
pf = pf.loc[:, ~(pf.iloc[-1, :] < 19000)]
# 删除指定列，包括
pf.drop(['信号与系统实验'], axis=1, inplace=True)
# 删除指定行，包括
pf.drop(['Row_sum'], axis=0, inplace=True)
# 将学号列放在第一列
student_number = pf['学号']
pf.drop(labels=['学号'], axis=1, inplace=True)
pf.insert(0, '学号', student_number)
pf = pf.reset_index(drop=True)
# 异常值处理
pf.drop([9], axis=0, inplace=True)
pf = pf.reset_index(drop=True)

# 将0值和nan值都用那一列的平均值代替
pf[pf == 0] = np.NaN
for column in list(pf.columns[pf.isnull().sum() > 0]):
    mean_val = pf[column].mean()
    pf[column].fillna(mean_val, inplace=True)

# 输出有意义的实验结果
pf.to_excel('grade_meaning.xls')

# 去掉学号列
pf.drop(labels=['学号'], axis=1, inplace=True)
# 对某一门课程可以进行聚类分析, 此处对c语言程序设计与飞行力学与导引两门课进行分析
grade_mining = pf['C语言程序设计']
grade_mining.to_excel('c.xls')

# 使用k-means进行聚类分析
k = 6  # 聚类的类别
iteration = 500  # 聚类的最大循环数
grade_mining_zs = 1.0 * (grade_mining - grade_mining.mean()) / grade_mining.std()  # 数据标准化
model = KMeans(n_clusters=k, n_jobs=4, max_iter=iteration)
grade_mining_zss = np.array(grade_mining_zs).reshape(-1, 1)
model.fit(grade_mining_zss)

# 简单打印结果
r1 = pd.Series(model.labels_).value_counts()        # 每一个聚类的数目
r2 = pd.DataFrame(model.cluster_centers_)           # 标准化的数据
r3 = r2 * grade_mining.std() + grade_mining.mean()  # 原始的数据成绩
r = pd.concat([r2, r3, r1], axis=1)
r.columns = [u'聚类中心(标准化)'] + [u'聚类中心(成绩)'] + [u'类别数目']
print(r)

# 详细输出原始数据及其类别
r_detail = pd.concat([grade_mining, pd.Series(model.labels_, index=grade_mining.index)], axis=1)
r_detail.columns = [u'成绩'] + ['聚类类别']
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
fig_output = '../tmp/c_programme'
for i in range(k):
    data_r = r_detail[r_detail['聚类类别'] == i].iloc[:, 0:]
    data = data_r.drop(labels=['聚类类别'], axis=1, inplace=False)
    density_plot(data).savefig(u'%s%s' % (fig_output, i))

# 评价这种聚类方法的优劣
# ch评价指标，值越大，聚类效果越好
grade_mining_np = np.array(grade_mining).reshape(-1, 1)
t = metrics.calinski_harabasz_score(grade_mining_np, model.labels_)
print("When cluster= {}\nThe calinski_harabasz_score= {}".format(k, t))
# 打印平均轮廓系数,平均轮廓系数最高,最好
s = silhouette_score(grade_mining_np, model.labels_)
print("When cluster= {}\nThe silhouette_score= {}".format(k, s))