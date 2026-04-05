from __future__ import annotations

import html
import pandas as pd
import streamlit as st


def _safe_monthly_df(df: pd.DataFrame) -> pd.DataFrame:
    if 'salary_unit' not in df.columns:
        return pd.DataFrame(columns=df.columns)
    out = df[df['salary_unit'].astype(str) == '月薪'].copy()
    if 'salary_avg' in out.columns:
        out['salary_avg'] = pd.to_numeric(out['salary_avg'], errors='coerce')
        out = out[out['salary_avg'].notna()].copy()
    return out


def render_kpi_row(items: list[tuple[str, str]]) -> None:
    cols = st.columns(len(items))
    for col, (label, value) in zip(cols, items):
        safe_label = html.escape(str(label))
        safe_value = html.escape(str(value))
        size_cls = ' small' if len(str(value)) >= 12 else ''
        col.markdown(
            f"""
            <div class='kpi-card'>
                <div class='kpi-label'>{safe_label}</div>
                <div class='kpi-value{size_cls}'>{safe_value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def build_overview_kpis(df: pd.DataFrame) -> list[tuple[str, str]]:
    monthly_df = _safe_monthly_df(df)
    city_count = df['city'].nunique() if 'city' in df.columns else 0
    keyword_count = df['keyword'].nunique() if 'keyword' in df.columns else 0
    avg_salary = monthly_df['salary_avg'].mean() if not monthly_df.empty else 0
    return [
        ('当前样本数', f'{len(df):,}'),
        ('月薪样本数', f'{len(monthly_df):,}'),
        ('覆盖城市数', f'{city_count:,}'),
        ('岗位类别数', f'{keyword_count:,}'),
        ('平均月薪', f'{avg_salary:,.0f} 元' if avg_salary else '暂无'),
        ('核心岗位数', '5'),
    ]
