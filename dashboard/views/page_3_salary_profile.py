from __future__ import annotations

import pandas as pd
import streamlit as st

from utils.data_loader import get_dataset
from utils.charts import plot_horizontal_bar, plot_bar, plot_donut, plot_box


EXP_ORDER = ['不限', '应届/在校', '1年以内', '1-3年', '3-5年', '5-10年', '10年以上', '其他/未说明']


def render(context: dict) -> None:
    filtered_df = context['filtered_df']
    datasets = context['datasets']

    st.markdown("<div class='page-title'>薪资画像页</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-desc'>展示岗位平均月薪、高薪阈值、薪资分布以及经验要求对薪资水平的影响。</div>", unsafe_allow_html=True)

    if filtered_df.empty:
        st.warning('当前筛选下没有数据。')
        return

    monthly_df = filtered_df[filtered_df['salary_unit'].astype(str) == '月薪'].copy() if 'salary_unit' in filtered_df.columns else pd.DataFrame()
    if 'salary_avg' in monthly_df.columns:
        monthly_df['salary_avg'] = pd.to_numeric(monthly_df['salary_avg'], errors='coerce')
        monthly_df = monthly_df[monthly_df['salary_avg'].notna()].copy()

    info_df = get_dataset(datasets, 'high_salary_info')
    threshold = info_df.loc[info_df['metric'] == 'high_salary_threshold_q75', 'value'] if not info_df.empty else None
    high_n = info_df.loc[info_df['metric'] == 'high_salary_count', 'value'] if not info_df.empty else None
    high_rate = info_df.loc[info_df['metric'] == 'high_salary_rate', 'value'] if not info_df.empty else None

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"<div class='kpi-card'><div class='kpi-label'>高薪阈值</div><div class='kpi-value'>{float(threshold.iloc[0]):,.0f} 元/月</div></div>" if threshold is not None and not threshold.empty else "<div class='kpi-card'><div class='kpi-label'>高薪阈值</div><div class='kpi-value'>暂无</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='kpi-card'><div class='kpi-label'>高薪岗位数</div><div class='kpi-value'>{int(float(high_n.iloc[0])):,}</div></div>" if high_n is not None and not high_n.empty else "<div class='kpi-card'><div class='kpi-label'>高薪岗位数</div><div class='kpi-value'>暂无</div></div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div class='kpi-card'><div class='kpi-label'>高薪占比</div><div class='kpi-value'>{float(high_rate.iloc[0])*100:.1f}%</div></div>" if high_rate is not None and not high_rate.empty else "<div class='kpi-card'><div class='kpi-label'>高薪占比</div><div class='kpi-value'>暂无</div></div>", unsafe_allow_html=True)

    c4, c5 = st.columns([1.05, 0.95])
    with c4:
        salary_mean_df = monthly_df.groupby('keyword', as_index=False)['salary_avg'].mean().sort_values('salary_avg', ascending=True)
        salary_mean_df = salary_mean_df.rename(columns={'salary_avg': 'mean'})
        plot_horizontal_bar(salary_mean_df, y='keyword', x='mean', title='不同岗位平均月薪')
    with c5:
        if not monthly_df.empty:
            above = int((monthly_df['salary_avg'] >= float(threshold.iloc[0])).sum()) if threshold is not None and not threshold.empty else 0
            below = max(len(monthly_df) - above, 0)
            plot_donut(['高薪岗位', '非高薪岗位'], [above, below], '高薪岗位占比')
        else:
            st.info('暂无月薪样本。')

    c6, c7 = st.columns(2)
    with c6:
        keyword_order = monthly_df.groupby('keyword')['salary_avg'].median().sort_values(ascending=False).index.tolist() if not monthly_df.empty else []
        plot_box(monthly_df, x='keyword', y='salary_avg', title='不同岗位月薪分布箱线图', category_orders={'keyword': keyword_order} if keyword_order else None)
    with c7:
        exp_order = [x for x in EXP_ORDER if x in monthly_df['experience_std'].dropna().astype(str).unique().tolist()] if not monthly_df.empty else []
        plot_box(monthly_df, x='experience_std', y='salary_avg', title='不同经验要求月薪分布箱线图', category_orders={'experience_std': exp_order} if exp_order else None)

    exp_salary_df = monthly_df.groupby('experience_std', as_index=False)['salary_avg'].mean().sort_values('salary_avg', ascending=False)
    exp_salary_df = exp_salary_df.rename(columns={'salary_avg': 'mean'})
    plot_bar(exp_salary_df, x='experience_std', y='mean', title='不同经验要求平均月薪')
