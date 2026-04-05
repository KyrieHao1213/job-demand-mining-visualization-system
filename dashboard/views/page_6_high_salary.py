from __future__ import annotations

import re
import streamlit as st

from config import HIGH_SALARY_KEYWORD_TABS
from utils.data_loader import get_dataset
from utils.charts import plot_bar, plot_diverging_bar, plot_horizontal_bar


def _clean_term_label(term: str) -> str:
    term = re.sub(r"C\((.*?)\)\[T\.(.*?)\]", r"\1: \2", str(term))
    replacements = {
        "experience_group, Treatment(reference='1-3年')": '经验',
        "city_tier, Treatment(reference='其他')": '城市层级',
        "degree_group, Treatment(reference='本科')": '学历',
        "company_size_model, Treatment(reference='微型或小型')": '公司规模',
        "keyword_model, Treatment(reference='数据分析师')": '岗位类别',
        'skill_python': 'Python',
        'skill_sql': 'SQL',
        'skill_excel': 'Excel',
    }
    for old, new in replacements.items():
        term = term.replace(old, new)
    return term


def render(context: dict) -> None:
    datasets = context['datasets']
    filters = context['filters']

    st.markdown("<div class='page-title'>高薪岗位特征页</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-desc'>综合展示高薪定义、结构化差异、Logistic 回归显著变量以及高薪差异词，是整套系统中最适合做研究结论收束的一页。</div>", unsafe_allow_html=True)

    info_df = get_dataset(datasets, 'high_salary_info')
    city_df = get_dataset(datasets, 'high_salary_city')
    company_df = get_dataset(datasets, 'high_salary_company')
    logit_df = get_dataset(datasets, 'logit_or')
    overall_diff = get_dataset(datasets, 'overall_diff')
    data_diff = get_dataset(datasets, 'data_diff')
    biz_diff = get_dataset(datasets, 'biz_diff')

    c1, c2, c3 = st.columns(3)
    with c1:
        threshold = info_df.loc[info_df['metric'] == 'high_salary_threshold_q75', 'value'] if not info_df.empty else None
        st.markdown(f"<div class='kpi-card'><div class='kpi-label'>高薪阈值</div><div class='kpi-value'>{float(threshold.iloc[0]):,.0f} 元/月</div></div>" if threshold is not None and not threshold.empty else "<div class='kpi-card'><div class='kpi-label'>高薪阈值</div><div class='kpi-value'>暂无</div></div>", unsafe_allow_html=True)
    with c2:
        high_n = info_df.loc[info_df['metric'] == 'high_salary_count', 'value'] if not info_df.empty else None
        st.markdown(f"<div class='kpi-card'><div class='kpi-label'>高薪岗位数</div><div class='kpi-value'>{int(float(high_n.iloc[0])):,}</div></div>" if high_n is not None and not high_n.empty else "<div class='kpi-card'><div class='kpi-label'>高薪岗位数</div><div class='kpi-value'>暂无</div></div>", unsafe_allow_html=True)
    with c3:
        high_rate = info_df.loc[info_df['metric'] == 'high_salary_rate', 'value'] if not info_df.empty else None
        st.markdown(f"<div class='kpi-card'><div class='kpi-label'>高薪占比</div><div class='kpi-value'>{float(high_rate.iloc[0]) * 100:.1f}%</div></div>" if high_rate is not None and not high_rate.empty else "<div class='kpi-card'><div class='kpi-label'>高薪占比</div><div class='kpi-value'>暂无</div></div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class='info-panel' style='margin-bottom:1rem;'>
          <div class='section-title'>Logistic 回归参考组说明</div>
          <div class='page-desc' style='margin-bottom:0;'>岗位类别参考组为<strong>数据分析师</strong>，学历参考组为<strong>本科</strong>，经验参考组为<strong>1-3年</strong>，城市层级参考组为<strong>其他城市</strong>，公司规模参考组为<strong>微型或小型公司</strong>。因此 OR 大于 1 表示相对于上述参考组更容易进入高薪岗位。</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c4, c5 = st.columns(2)
    with c4:
        plot_bar(city_df, x='city_tier', y='rate_in_group', color='salary_group', title='高薪组与非高薪组的城市层级分布')
    with c5:
        plot_bar(company_df, x='company_size_tier', y='rate_in_group', color='salary_group', title='高薪组与非高薪组的公司规模分布')

    c6, c7 = st.columns([1.0, 1.25])
    with c6:
        if not logit_df.empty:
            sub = logit_df[logit_df['term'] != 'Intercept'].copy()
            if 'p_value' in sub.columns:
                sub = sub[sub['p_value'] < 0.05].copy()
            sub['term_clean'] = sub['term'].map(_clean_term_label)
            sub = sub.sort_values('odds_ratio', ascending=True)
            plot_horizontal_bar(sub, y='term_clean', x='odds_ratio', title='Logistic 回归显著变量 OR')
    with c7:
        plot_diverging_bar(overall_diff, term_col='term', value_col='diff', side_col='side', title='整体高薪差异词')

    selected_keywords = filters.get('keyword', [])
    selected_keywords = [kw for kw in selected_keywords if kw in HIGH_SALARY_KEYWORD_TABS]
    if len(selected_keywords) == 1:
        focus_keyword = selected_keywords[0]
        st.markdown(f"<div class='note-box'>当前已根据全局筛选自动聚焦到：<strong>{focus_keyword}</strong></div>", unsafe_allow_html=True)
    else:
        default_idx = HIGH_SALARY_KEYWORD_TABS.index(selected_keywords[0]) if selected_keywords else 0
        focus_keyword = st.radio('选择重点岗位', HIGH_SALARY_KEYWORD_TABS, horizontal=True, index=default_idx)
        if len(selected_keywords) > 1:
            st.markdown("<div class='note-box'>当前全局筛选包含多个重点岗位，因此这里保留页内二次选择。</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>重点岗位内部差异词</div>", unsafe_allow_html=True)
    df = data_diff if focus_keyword == '数据分析师' else biz_diff
    plot_diverging_bar(df, term_col='term', value_col='diff', side_col='side', title=f'{focus_keyword}内部高薪差异词')

    st.markdown("<div class='note-box'><strong>说明：</strong> 本页刻意不展示“高薪主题差异分析”，与论文正文的最终收口保持一致。</div>", unsafe_allow_html=True)
