
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import PRIMARY_COLOR, SECONDARY_COLOR, ACCENT_COLOR, SUCCESS_COLOR, SALARY_GROUP_COLOR_MAP

BASE_SEQUENCE = [PRIMARY_COLOR, SECONDARY_COLOR, ACCENT_COLOR, SUCCESS_COLOR, '#8B5CF6', '#14B8A6', '#EF4444']


def empty_hint(message: str = '暂无可展示数据') -> None:
    st.info(message)



def _base_layout(fig, height: int = 380):
    fig.update_layout(
        template='plotly_white',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=12, r=12, t=56, b=12),
        height=height,
        legend_title_text='',
        font=dict(size=13, color='#334155'),
        title=dict(font=dict(size=18, color='#1f2a44')),
    )
    fig.update_xaxes(showgrid=True, gridcolor='#E8EEF7', zeroline=False)
    fig.update_yaxes(showgrid=False, zeroline=False)
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})



def plot_bar(df: pd.DataFrame, x: str, y: str, title: str, color: str | None = None, orientation: str = 'v'):
    if df.empty:
        empty_hint()
        return
    color_map = SALARY_GROUP_COLOR_MAP if color == 'salary_group' else None
    fig = px.bar(
        df,
        x=x,
        y=y,
        color=color,
        orientation=orientation,
        title=title,
        color_discrete_sequence=BASE_SEQUENCE,
        color_discrete_map=color_map,
    )
    _base_layout(fig, height=390)



def plot_horizontal_bar(df: pd.DataFrame, y: str, x: str, title: str, color: str | None = None):
    if df.empty:
        empty_hint()
        return
    color_map = SALARY_GROUP_COLOR_MAP if color == 'salary_group' else None
    fig = px.bar(
        df,
        y=y,
        x=x,
        color=color,
        orientation='h',
        title=title,
        color_discrete_sequence=BASE_SEQUENCE,
        color_discrete_map=color_map,
    )
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    _base_layout(fig, height=430)



def plot_stacked_bar(df: pd.DataFrame, x: str, y: str, color: str, title: str):
    if df.empty:
        empty_hint()
        return
    fig = px.bar(
        df,
        x=x,
        y=y,
        color=color,
        title=title,
        color_discrete_sequence=BASE_SEQUENCE,
    )
    fig.update_layout(barmode='stack')
    _base_layout(fig, height=430)



def plot_heatmap(
    df: pd.DataFrame,
    x_cols: list[str],
    y_col: str,
    title: str,
    *,
    value_format: str = 'count',
    colorscale: list | None = None,
):
    if df.empty or y_col not in df.columns:
        empty_hint()
        return

    heat_df = df[[y_col] + x_cols].copy().set_index(y_col)
    z = heat_df.values

    if value_format == 'percent':
        text = [[f'{v:.1%}' for v in row] for row in z]
        hovertemplate = '岗位: %{y}<br>技能: %{x}<br>岗位内占比: %{z:.1%}<extra></extra>'
        zmin, zmax = 0, float(z.max()) if len(z) else 1
    else:
        text = [[f'{int(v)}' for v in row] for row in z]
        hovertemplate = '岗位: %{y}<br>技能: %{x}<br>出现次数: %{z}<extra></extra>'
        zmin, zmax = 0, float(z.max()) if len(z) else 1

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=heat_df.columns,
        y=heat_df.index,
        text=text,
        texttemplate='%{text}',
        textfont={'size': 11},
        colorscale=colorscale or [[0, '#F4F8FF'], [1, PRIMARY_COLOR]],
        zmin=zmin,
        zmax=zmax,
        hovertemplate=hovertemplate,
        hoverongaps=False,
        colorbar=dict(title='占比' if value_format == 'percent' else '次数'),
    ))
    fig.update_xaxes(side='bottom')
    fig.update_layout(title=title)
    _base_layout(fig, height=440)



def plot_line(df: pd.DataFrame, x: str, y: str, title: str):
    if df.empty:
        empty_hint()
        return
    fig = px.line(df, x=x, y=y, markers=True, title=title)
    fig.update_traces(line=dict(color=PRIMARY_COLOR, width=3), marker=dict(color=PRIMARY_COLOR, size=9))
    _base_layout(fig, height=360)



def plot_donut(labels: list[str], values: list[float], title: str):
    if not labels or not values:
        empty_hint()
        return
    fig = go.Figure(
        data=[go.Pie(labels=labels, values=values, hole=0.62, marker=dict(colors=BASE_SEQUENCE[:len(labels)]))]
    )
    fig.update_traces(textinfo='percent', textfont_size=13)
    fig.update_layout(title=title)
    _base_layout(fig, height=360)



def plot_box(df: pd.DataFrame, x: str, y: str, title: str, color: str | None = None, category_orders: dict | None = None):
    if df.empty:
        empty_hint()
        return
    color_map = SALARY_GROUP_COLOR_MAP if color == 'salary_group' else None
    fig = px.box(
        df,
        x=x,
        y=y,
        color=color,
        title=title,
        color_discrete_sequence=BASE_SEQUENCE,
        color_discrete_map=color_map,
        category_orders=category_orders,
        points='outliers',
    )
    _base_layout(fig, height=430)



def plot_diverging_bar(df: pd.DataFrame, term_col: str, value_col: str, side_col: str, title: str):
    if df.empty:
        empty_hint()
        return
    plot_df = df.copy()
    if side_col in plot_df.columns:
        plot_df['display_value'] = plot_df.apply(
            lambda r: r[value_col] if str(r[side_col]) == '高薪更突出' else r[value_col],
            axis=1,
        )
        fig = px.bar(
            plot_df.sort_values('display_value'),
            x='display_value',
            y=term_col,
            color=side_col,
            orientation='h',
            color_discrete_map=SALARY_GROUP_COLOR_MAP | {'高薪更突出': ACCENT_COLOR, '非高薪更突出': PRIMARY_COLOR},
            title=title,
        )
    else:
        fig = px.bar(plot_df, x=value_col, y=term_col, orientation='h', title=title, color_discrete_sequence=BASE_SEQUENCE)
    fig.update_layout(yaxis={'categoryorder': 'array', 'categoryarray': plot_df.sort_values('display_value')[term_col] if 'display_value' in plot_df else None})
    _base_layout(fig, height=470)
