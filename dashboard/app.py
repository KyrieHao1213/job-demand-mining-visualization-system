from __future__ import annotations

from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import streamlit as st

from config import APP_TITLE, APP_SUBTITLE, LAYOUT, PAGE_ICON
from utils.data_loader import load_all_datasets, get_dataset
from utils.filters import render_sidebar_filters, apply_filters, render_active_filters
from utils.kpi import build_overview_kpis, render_kpi_row
from views import (
    page_1_overview,
    page_2_job_profile,
    page_3_salary_profile,
    page_4_skills,
    page_5_text_mining,
    page_6_high_salary,
)

st.set_page_config(page_title=APP_TITLE, page_icon=PAGE_ICON, layout=LAYOUT)

PAGE_REGISTRY = {
    '总览页': page_1_overview.render,
    '岗位画像页': page_2_job_profile.render,
    '薪资画像页': page_3_salary_profile.render,
    '技能需求页': page_4_skills.render,
    '文本挖掘页': page_5_text_mining.render,
    '高薪岗位特征页': page_6_high_salary.render,
}


def load_css() -> None:
    css_path = Path(__file__).resolve().parent / 'assets' / 'style.css'
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def _hero_hint() -> str:
    return '展示数据分析岗位需求的可视化系统，统一整合岗位画像、技能需求、文本挖掘与高薪岗位特征。'


def init_data() -> None:
    datasets = load_all_datasets()
    base_df = get_dataset(datasets, 'base_jobs')
    st.session_state['datasets'] = datasets
    st.session_state['base_jobs'] = base_df
    if base_df.empty:
        st.error('未读取到基础数据 clean_jobs_filtered.csv，请先确认数据路径。')
        st.stop()



def render_shell() -> tuple[str, dict]:
    datasets = st.session_state['datasets']
    base_df = st.session_state['base_jobs']

    page_name = st.sidebar.radio('选择页面', list(PAGE_REGISTRY.keys()), key='current_page')
    filters = render_sidebar_filters(base_df)
    filtered_df = apply_filters(base_df, filters)

    st.session_state['filtered_jobs'] = filtered_df
    st.session_state['global_filters'] = filters

    st.markdown(
        f"""
        <div class='hero-box'>
            <div class='main-title'>{APP_TITLE}</div>
            <div class='sub-title'>{APP_SUBTITLE}</div>
            <div class='section-note'>{_hero_hint()}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_active_filters(filters)
    render_kpi_row(build_overview_kpis(filtered_df))
    st.markdown('---')

    context = {
        'datasets': datasets,
        'base_df': base_df,
        'filtered_df': filtered_df,
        'filters': filters,
        'page_name': page_name,
    }
    return page_name, context


load_css()
init_data()
page_name, context = render_shell()
PAGE_REGISTRY[page_name](context)
