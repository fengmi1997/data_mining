import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

# 数据挖掘大作业
# by 翟艺伟，学号：2018201653

# 读取excel
pf = pd.read_excel('grade_original.xls', sheet_name='Sheet0')
# 筛选出10-14级同学的数据用作数据挖掘
pf.drop(['学年度', '学期', '课程序号', '课程代码', '授课教师', '实得学分', '绩点', '课程类别代码', '修读类别', '学生类别', '课程学分'], axis=1, inplace=True)
pf = pf[pf['班级'].isin(['08031401', '08031402', '08031403', '08031301', '08031302', '08031201', '08031202', '08031101',
                       '08031102', '08031001', '08031002'])]
pf.drop(['班级'], axis=1, inplace=True)
# 去掉这些班里面上一级14级的数据
pf = pf[pf['学号'] > 2010300000]
# 对13级的英语课程做特殊处理
pf.loc[pf[(pf['课程名称'] == '大学英语(2)')].index,['课程类别']]= '必修'
pf.loc[pf[(pf['课程名称'] == '大学英语(1)')].index,['课程类别']]= '必修'
pf.loc[pf[(pf['课程名称'] == '经典控制原理I')].index,['课程名称']]= '自动控制原理'
pf.loc[pf[(pf['课程名称'] == '经典控制课程设计')].index,['课程名称']]= '自动控制原理课程设计'
# 去掉课程类别为任选的课
pf = pf[~pf['课程类别'].isin(['专业课选修课', '通识通修', '综合素养', '科学素养类课程', '艺术素养课程', '艺术素养类课程', '人文素养类课程', '综合素质教育课程课组', '任选'])]
pf = pf[~pf['课程名称'].isin(['研究训练', '大学生心理健康教育', '大学生职业生涯规划', '航空概论',
                          '航天概论', '航海概论', 'Matlab仿真设计', 'Matlab仿真通信', '上机训练',
                          '不确定信息融合推理', '人文阅读赏析', '人机接口设计', '人民军队历史与优良传统(国防生)',
                          '人民军队导论', '传感器原理与应用', '信号与系统 I', '信号与系统 I.1', '信号与系统.1',
                          '信息隐藏技术', '光电技术', '军事思想', '军事技能训练', '军事法概论', '军事领导科学与方法',
                          '军人心理学', '军训', '军队基层管理', '创新性综合实验', 'C程序设计II实验.1', 'DSP/FPGA原理及应用',
                          '世界政治经济学与国际关系'])]
# 删除体育类课程
pf = pf[~pf["课程名称"].str.contains("体育")]
# 删除形式与政策
pf = pf[~pf["课程名称"].str.contains("形式与政策")]
# 在课程类别列里面将‘C程序设计II’转化为‘C语言程序设计’，‘C程序设计II实验’转化为‘C语言程序设计实验’
pf.ix[pf['课程名称'] == 'C程序设计II', '课程名称'] = 'C语言程序设计'
pf.ix[pf['课程名称'] == 'C程序设计II实验', '课程名称'] = 'C语言程序设计实验'
# 在课程类别列里面将大学物理一致化
pf.ix[pf['课程名称'] == '大学物理II（上）', '课程名称'] = '大学物理（1）'
pf.ix[pf['课程名称'] == '大学物理II（下）', '课程名称'] = '大学物理（2）'
pf.ix[pf['课程名称'] == '大学物理I（上）', '课程名称'] = '大学物理（1）'
pf.ix[pf['课程名称'] == '大学物理I（下）', '课程名称'] = '大学物理（2）'
pf.ix[pf['课程名称'] == '大学物理（下）', '课程名称'] = '大学物理（2）'
pf.to_excel('drop.xls')
# 在课程类别列里面将大学英语一致化
pf.ix[pf['课程名称'] == '大学英语(1)', '课程名称'] = '大学英语（1）'
pf.ix[pf['课程名称'] == '大学英语(2)', '课程名称'] = '大学英语（2）'
pf.ix[pf['课程名称'] == '大学英语一级', '课程名称'] = '大学英语（1）'
pf.ix[pf['课程名称'] == '大学英语三级', '课程名称'] = '大学英语（2）'
pf.ix[pf['课程名称'] == '大学英语（Ⅰ）', '课程名称'] = '大学英语（1）'
pf.ix[pf['课程名称'] == '大学英语（Ⅱ）', '课程名称'] = '大学英语（2）'
# 综合实验
pf.ix[pf['课程名称'] == '探测指导与控制专业综合实验开放实验', '课程名称'] = '探测制导与控制专业综合实验开放实验'
pf.ix[pf['课程名称'] == '探测制导与控制技术专业实验', '课程名称'] = '探测制导与控制专业综合实验开放实验'
# 机械制图
pf.ix[pf['课程名称'] == '机械制图基础', '课程名称'] = '机械制图'
# 将机载探测与电子对抗原理和机载探测系统合并
pf.ix[pf['课程名称'] == '机载探测与电子对抗原理', '课程名称'] = '机载探测系统'
# 现代控制理论
pf.ix[pf['课程名称'] == '现代控制理论（双语）', '课程名称'] = '现代控制理论'
pf.ix[pf['课程名称'] == '现代控制理论（英）', '课程名称'] = '现代控制理论'
# 理论力学
pf.ix[pf['课程名称'] == '理论力学Ⅱ', '课程名称'] = '理论力学'
pf.ix[pf['课程名称'] == '理论力学Ⅲ', '课程名称'] = '理论力学'
# 电子实习
pf.ix[pf['课程名称'] == '电子实习A', '课程名称'] = '电子实习'
# 电路基础
pf.ix[pf['课程名称'] == '电路分析基础', '课程名称'] = '电路基础'
pf.ix[pf['课程名称'] == '电路分析基础 I', '课程名称'] = '电路基础'
pf.ix[pf['课程名称'] == '电路基础 I', '课程名称'] = '电路基础'
pf.ix[pf['课程名称'] == '电路分析基础 I 实验', '课程名称'] = '电路基础实验'
pf.ix[pf['课程名称'] == '电路分析基础实验', '课程名称'] = '电路基础实验'
pf.ix[pf['课程名称'] == '电路基础 I 实验（英）', '课程名称'] = '电路基础实验'
# 线性代数
pf.ix[pf['课程名称'] == '线性代数Ⅰ', '课程名称'] = '线性代数'
pf.ix[pf['课程名称'] == '线性代数Ⅱ', '课程名称'] = '线性代数'
# 综合航电
pf.ix[pf['课程名称'] == '航空电子系统综合化', '课程名称'] = '综合航空电子系统'
# 航空电子系统
pf.ix[pf['课程名称'] == '航空电子系统进展（英）', '课程名称'] = '航空电子系统进展'
pf.ix[pf['课程名称'] == '航空电子系统（英）', '课程名称'] = '航空电子系统进展'
pf.ix[pf['课程名称'] == '航空电子系统（英语）', '课程名称'] = '航空电子系统进展'
# 飞行力学与导引与航空系统概论，飞行器导论
pf.ix[pf['课程名称'] == '航空系统概论', '课程名称'] = '飞行力学与导引'
pf.ix[pf['课程名称'] == '飞行器导论', '课程名称'] = '飞行力学与导引'
# 高数
pf.ix[pf['课程名称'] == '高等数学（上）', '课程名称'] = '高等数学（1）'
pf.ix[pf['课程名称'] == '高等数学（下）', '课程名称'] = '高等数学（2）'

# pf.sort_values(by=["学号"], inplace=True, ascending=[True])
pf.drop(['课程类别'], axis=1, inplace=True)

# pf.to_excel('drop.xls')

id_year_group = (pf.groupby('学号'))
id_curriculum = pd.DataFrame()
id_curriculum.to_excel('id_curriculum.xls')

for group_name, group_data in id_year_group:
    # print(group_name)
    group_data.drop(['学号'], axis=1, inplace=True)
    # group_data.drop([group_data.columns[0]], axis=1, inplace=True)  # Note: zero indexed
    # print(group_data)
    reshape_data = np.array(group_data).transpose()
    group_reshape = pd.DataFrame(reshape_data)
    group_reshape.rename(columns=group_reshape.ix[0], inplace=True)
    group_reshape.drop(0, axis=0, inplace=True)
    group_reshape.reset_index(drop=True, inplace=True)
    id_frame = pd.DataFrame({'学号': [group_name]})
    id_each_curriculum = pd.concat([id_frame, group_reshape], axis=1)

    id_each_curriculum.to_excel('id_each_curriculum.xls')

    pf1 = pd.read_excel('id_curriculum.xls', sheet_name='Sheet1')
    pf2 = pd.read_excel('id_each_curriculum.xls', sheet_name='Sheet1')
    id_curriculum = pd.concat([pf1, pf2], axis=0)
    id_curriculum.to_excel('id_curriculum.xls')

    # id_curriculum = pd.concat([id_curriculum, id_each_curriculum], axis=0)
    id_each_curriculum = pd.DataFrame()
    # print('-----')


