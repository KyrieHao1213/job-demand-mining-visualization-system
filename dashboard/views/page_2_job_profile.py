from __future__ import annotations

import pandas as pd
import streamlit as st

from config import CORE_KEYWORDS
from utils.charts import plot_donut, plot_stacked_bar


def render(context: dict) -> None:
    filtered_df = context['filtered_df']

    st.markdown("<div class='page-title'>岗位画像页</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-desc'>聚焦五类核心岗位，观察其在城市、学历与经验要求上的结构性差异。</div>", unsafe_allow_html=True)

    if filtered_df.empty or 'keyword' not in filtered_df.columns:
        st.warning('当前筛选下没有数据。')
        return

    core_df = filtered_df[filtered_df['keyword'].isin(CORE_KEYWORDS)].copy()
    if core_df.empty:
        st.info('当前筛选下未命中五类核心岗位。')
        return

    share_df = core_df['keyword'].value_counts().reindex(CORE_KEYWORDS).dropna().reset_index()
    share_df.columns = ['keyword', 'count']

    top_cities = core_df['city'].value_counts().head(8).index.tolist()
    city_stack = core_df[core_df['city'].isin(top_cities)].groupby(['keyword', 'city']).size().reset_index(name='count')
    degree_stack = core_df.groupby(['keyword', 'degree_std']).size().reset_index(name='count')
    exp_stack = core_df.groupby(['keyword', 'experience_std']).size().reset_index(name='count')

    c1, c2 = st.columns([0.9, 1.3])
    with c1:
        plot_donut(share_df['keyword'].tolist(), share_df['count'].tolist(), '五类核心岗位占比')
    with c2:
        plot_stacked_bar(city_stack, x='keyword', y='count', color='city', title='岗位 × 城市分布（Top8 城市）')

    c3, c4 = st.columns(2)
    with c3:
        plot_stacked_bar(degree_stack, x='keyword', y='count', color='degree_std', title='岗位 × 学历分布')
    with c4:
        plot_stacked_bar(exp_stack, x='keyword', y='count', color='experience_std', title='岗位 × 经验分布')

    st.markdown(
        f"<div class='note-box'><strong>页面结论：</strong> 数据分析岗位并不是单一职位，而是已经形成了数据分析师、BI 分析师、用户分析师、经营分析师、商业分析师等多类细分方向；不同方向在城市集聚、学历门槛与经验要求上呈现出明显差异。</div>",
        unsafe_allow_html=True,
    )
