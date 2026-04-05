
from __future__ import annotations

import pandas as pd
import streamlit as st

from config import SAMPLE_SCOPE_INFO
from utils.charts import plot_horizontal_bar, plot_bar, plot_donut


def render(context: dict) -> None:
    filtered_df = context['filtered_df']

    st.markdown("<div class='page-title'>总览页</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-desc'>从总体样本规模、岗位分布、城市集聚以及学历经验门槛四个维度快速把握当前招聘市场特征。</div>", unsafe_allow_html=True)

    if filtered_df.empty:
        st.warning('当前筛选下没有数据。')
        return

    st.markdown(
        f"""
        <div class='info-panel'>
          <div class='section-title'>数据口径说明</div>
          <ul>
            <li><strong>原始抓取样本：</strong>{SAMPLE_SCOPE_INFO['raw_total']} 条</li>
            <li><strong>标题过滤后有效样本：</strong>{SAMPLE_SCOPE_INFO['filtered_total']} 条</li>
            <li><strong>有效月薪样本：</strong>{SAMPLE_SCOPE_INFO['monthly_total']} 条</li>
            <li><strong>文本挖掘对象：</strong>{SAMPLE_SCOPE_INFO['core_keyword_desc']}</li>
            <li><strong>高薪阈值：</strong>{SAMPLE_SCOPE_INFO['high_salary_threshold']}（按全部有效月薪样本 75 分位数定义）</li>
          </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    keyword_df = filtered_df['keyword'].value_counts().reset_index()
    keyword_df.columns = ['keyword', 'count']
    city_df = filtered_df['city'].value_counts().head(15).reset_index()
    city_df.columns = ['city', 'count']
    degree_df = filtered_df['degree_std'].value_counts().reset_index()
    degree_df.columns = ['degree_std', 'count']
    exp_df = filtered_df['experience_std'].value_counts().reset_index()
    exp_df.columns = ['experience_std', 'count']

    monthly_n = int((filtered_df['salary_unit'].astype(str) == '月薪').sum()) if 'salary_unit' in filtered_df.columns else 0
    other_n = max(len(filtered_df) - monthly_n, 0)

    c1, c2, c3 = st.columns([1.05, 1.05, 0.9])
    with c1:
        plot_horizontal_bar(keyword_df, y='keyword', x='count', title='岗位类别分布')
    with c2:
        plot_horizontal_bar(city_df, y='city', x='count', title='城市 Top15 分布')
    with c3:
        plot_donut(['月薪样本', '其他'], [monthly_n, other_n], '月薪样本占比')

    c4, c5 = st.columns(2)
    with c4:
        plot_bar(degree_df, x='degree_std', y='count', title='学历分布')
    with c5:
        plot_bar(exp_df, x='experience_std', y='count', title='经验分布')

    top_city = city_df.iloc[0]['city'] if not city_df.empty else '暂无'
    top_kw = keyword_df.iloc[0]['keyword'] if not keyword_df.empty else '暂无'
    st.markdown(
        f"<div class='note-box'><strong>页面结论：</strong> 当前筛选条件下，招聘需求最集中的城市为 <strong>{top_city}</strong>，最主要的岗位类别为 <strong>{top_kw}</strong>。该页适合在答辩开场快速说明样本结构与市场总体格局。</div>",
        unsafe_allow_html=True,
    )
