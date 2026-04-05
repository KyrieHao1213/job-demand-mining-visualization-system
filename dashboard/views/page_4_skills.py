
from __future__ import annotations

from collections import Counter

import pandas as pd
import streamlit as st

from utils.charts import plot_horizontal_bar, plot_heatmap


def _skill_counter(series: pd.Series) -> Counter:
    counter = Counter()
    for value in series.dropna():
        for item in [x.strip() for x in str(value).split('|') if str(x).strip()]:
            counter[item] += 1
    return counter



def _build_skill_share_matrix(df: pd.DataFrame, keyword_col: str = 'keyword', skill_col: str = 'skill_keywords_extract', top_keywords: int = 8, top_skills: int = 10) -> pd.DataFrame:
    if keyword_col not in df.columns or skill_col not in df.columns:
        return pd.DataFrame()

    top_keyword_list = df[keyword_col].value_counts().head(top_keywords).index.tolist()
    if not top_keyword_list:
        return pd.DataFrame()

    skill_counter = Counter()
    for value in df[skill_col].dropna():
        for item in [x.strip() for x in str(value).split('|') if x.strip()]:
            skill_counter[item] += 1
    top_skill_list = [s for s, _ in skill_counter.most_common(top_skills)]
    if not top_skill_list:
        return pd.DataFrame()

    count_matrix = pd.DataFrame(0, index=top_keyword_list, columns=top_skill_list, dtype=float)
    keyword_sizes = df[df[keyword_col].isin(top_keyword_list)][keyword_col].value_counts()

    for _, row in df[[keyword_col, skill_col]].dropna().iterrows():
        kw = row[keyword_col]
        if kw not in top_keyword_list:
            continue
        skills = [x.strip() for x in str(row[skill_col]).split('|') if x.strip()]
        for skill in skills:
            if skill in top_skill_list:
                count_matrix.loc[kw, skill] += 1

    for kw in count_matrix.index:
        denom = float(keyword_sizes.get(kw, 0))
        if denom > 0:
            count_matrix.loc[kw, :] = count_matrix.loc[kw, :] / denom

    return count_matrix



def render(context: dict) -> None:
    filtered_df = context['filtered_df']

    st.markdown("<div class='page-title'>技能需求页</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-desc'>展示岗位通用技能、岗位差异技能以及基础工具层—可视化层—进阶建模层的三层能力结构。</div>", unsafe_allow_html=True)

    if filtered_df.empty:
        st.warning('当前筛选下没有数据。')
        return

    counter = _skill_counter(filtered_df.get('skill_keywords_extract', pd.Series(dtype=str)))
    skill_top20 = pd.DataFrame(counter.most_common(20), columns=['skill', 'count'])

    share_matrix = _build_skill_share_matrix(filtered_df)
    skill_heatmap = share_matrix.reset_index().rename(columns={'index': 'keyword'}) if not share_matrix.empty else pd.DataFrame()

    c1, c2 = st.columns([0.95, 1.25])
    with c1:
        plot_horizontal_bar(skill_top20, y='skill', x='count', title='技能 Top20')
    with c2:
        if not skill_heatmap.empty:
            x_cols = [c for c in skill_heatmap.columns if c != 'keyword']
            plot_heatmap(skill_heatmap, x_cols=x_cols, y_col='keyword', title='岗位 × 核心技能占比热力图', value_format='percent')
            st.caption('说明：为避免大样本岗位带偏，本图采用“岗位内提及该技能的职位占比”而非绝对出现次数。')
        else:
            st.info('当前筛选下暂无可展示的技能热力图。')

    st.markdown(
        """
        <div class='info-panel'>
          <div class='section-title'>技能结构说明</div>
          <ul>
            <li><strong>基础分析工具层：</strong> SQL / Excel / Python</li>
            <li><strong>可视化与业务输出层：</strong> Tableau / Power BI / 数据可视化 / 报表开发</li>
            <li><strong>进阶建模与数据工程层：</strong> 机器学习 / 数据建模 / 数据仓库 / ETL 等</li>
          </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
