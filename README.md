

**# Job Demand Mining Visualization System**



基于招聘信息文本的数据分析岗位需求挖掘与可视化系统。



本项目围绕数据分析相关岗位的招聘信息展开，结合描述统计、关键词提取、LDA 主题模型和高薪岗位特征分析，构建了一套可交互的 Streamlit 可视化系统，用于展示岗位画像、技能需求、文本挖掘结果以及高薪岗位特征。



**## 项目内容**



系统主要包括以下几个模块：



\- 总览页：展示样本规模、岗位分布、城市分布、学历与经验结构

\- 岗位画像页：展示五类核心岗位在城市、学历、经验上的结构差异

\- 薪资画像页：展示岗位平均月薪、高薪阈值以及薪资分布特征

\- 技能需求页：展示通用技能、高频技能以及岗位内技能占比结构

\- 文本挖掘页：展示 TF-IDF、TextRank、LDA 主题模型及岗位-主题对应关系

\- 高薪岗位特征页：展示高薪岗位定义、结构化差异、Logistic 回归结果和高薪差异词



**## 研究对象**



本项目聚焦数据分析相关岗位，并在文本挖掘部分重点分析以下五类核心岗位：



\- 数据分析师

\- BI分析师

\- 用户分析师

\- 经营分析师

\- 商业分析师



**## 方法概览**



项目主要使用以下方法：



\- 描述统计分析

\- 技能关键词提取

\- TF-IDF 关键词分析

\- TextRank 关键词分析

\- LDA 主题模型

\- 高薪岗位特征分析

\- Logistic 回归解释模型

\- Streamlit 可视化展示



**## 项目结构**



```text

.

├── dashboard/                  # Streamlit 可视化系统

│   ├── app.py                  # 入口文件

│   ├── requirements.txt

│   ├── assets/

│   ├── utils/

│   └── views/                  # 各页面代码

├── data/

│   └── processed/

│       └── clean\_jobs\_filtered.csv

├── output/

│   ├── eda\_tables/

│   ├── keyword\_tables/

│   ├── lda\_tables/

│   ├── high\_salary\_tables/

│   └── high\_salary\_keyword\_tables/

├── clean\_jobs.py

├── eda\_analysis.py

├── keyword\_analysis.py

├── lda\_analysis.py

├── high\_salary\_analysis.py

└── high\_salary\_keyword\_diff.py



**## 数据说明**



系统展示依赖两类数据：



清洗后的底表

data/processed/clean\_jobs\_filtered.csv

各分析模块输出的结果表

位于 output/ 目录下，包括：

EDA 结果

关键词分析结果

LDA 主题模型结果

高薪岗位分析结果

高薪关键词差异结果



当前系统采用“离线分析 + 在线展示”的方式，页面不在前端实时重跑模型，而是直接读取已生成的结果文件。



**## 本地运行方式**



先安装依赖：

pip install -r dashboard/requirements.txt



然后启动系统：

streamlit run dashboard/app.py



**## 部署说明**



本项目可部署到 Streamlit Community Cloud。



部署时请注意：



入口文件为 dashboard/app.py

需要一并上传 data/processed/clean\_jobs\_filtered.csv

需要一并上传 output/ 下各分析结果表

依赖文件使用 dashboard/requirements.txt



**## 当前版本说明**



当前版本重点完成了：



六页可视化系统搭建

全局筛选与跨页面联动

文本挖掘页的关键词分析与主题分析展示

高薪岗位特征页的结构化差异、OR 结果与差异词展示

岗位 × 主题平均概率热力图

岗位内技能占比热力图



**## 说明**



本项目主要用于课程研究与可视化展示，重点在于将招聘信息文本分析结果转化为结构清晰、可交互的系统页面，方便对岗位需求特征进行直观观察和总结。

