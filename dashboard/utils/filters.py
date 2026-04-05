from __future__ import annotations

import pandas as pd
import streamlit as st

FILTER_FIELDS = [
    ('keyword', '岗位类别'),
    ('city', '城市'),
    ('degree_std', '学历'),
    ('experience_std', '经验'),
    ('company_size_raw', '公司规模'),
    ('salary_unit', '薪资单位'),
]
ALL_TOKEN = '全部'

STRUCTURED_ORDER = {
    'degree_std': ['不限', '大专', '本科', '硕士', '博士', '硕士及以上', '其他/未说明'],
    'experience_std': ['不限', '应届/在校', '1年以内', '应届及1年以内', '1-3年', '3-5年', '5-10年', '5年以上', '10年以上', '其他/未说明'],
    'company_size_raw': ['20人以下', '20-99人', '100-299人', '300-499人', '500-999人', '1000-9999人', '10000人以上', '其他/未说明'],
    'salary_unit': ['月薪', '年薪', '日薪'],
}

try:
    from pypinyin import lazy_pinyin  # type: ignore
except Exception:  # pragma: no cover
    lazy_pinyin = None


def _normalize_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        raw = value
    else:
        raw = [value]
    seen = []
    for v in raw:
        s = str(v).strip()
        if s and s not in seen:
            seen.append(s)
    return seen


def _sort_by_pinyin(values: list[str]) -> list[str]:
    if lazy_pinyin is None:
        return sorted(values)
    return sorted(values, key=lambda x: ''.join(lazy_pinyin(x)))


def _sort_with_structure(values: list[str], ordered_prefix: list[str]) -> list[str]:
    seen = set(values)
    ordered = [x for x in ordered_prefix if x in seen]
    remaining = [x for x in values if x not in ordered]
    return ordered + _sort_by_pinyin(remaining)


def _get_options(base_df: pd.DataFrame, col: str) -> list[str]:
    if col not in base_df.columns:
        return []
    vals = base_df[col].dropna().astype(str).str.strip()
    vals = vals[vals != '']
    unique_vals = vals.unique().tolist()

    if col in ('keyword', 'city'):
        return _sort_by_pinyin(unique_vals)
    if col in STRUCTURED_ORDER:
        return _sort_with_structure(unique_vals, STRUCTURED_ORDER[col])
    return _sort_by_pinyin(unique_vals)


def _normalize_widget_selection(value, valid_options: list[str]) -> list[str]:
    selected = _normalize_list(value)
    selected = [v for v in selected if v == ALL_TOKEN or v in valid_options]
    if not selected:
        return [ALL_TOKEN]
    return selected


def _widget_to_filter_values(widget_value, valid_options: list[str]) -> list[str]:
    selected = _normalize_widget_selection(widget_value, valid_options)
    if not selected or ALL_TOKEN in selected:
        return list(valid_options)
    return selected


def _widget_prev_key(col: str) -> str:
    return f'_prev_filter_{col}'


def _handle_multiselect_change(col: str) -> None:
    options_map = st.session_state.get('_filter_options_map', {})
    valid_options = list(options_map.get(col, []))
    widget_key = f'filter_{col}'
    prev_key = _widget_prev_key(col)

    previous = _normalize_list(st.session_state.get(prev_key, [ALL_TOKEN]))
    current = _normalize_widget_selection(st.session_state.get(widget_key, [ALL_TOKEN]), valid_options)

    if not current:
        current = [ALL_TOKEN]
    elif ALL_TOKEN in current and len(current) > 1:
        if ALL_TOKEN in previous:
            current = [v for v in current if v != ALL_TOKEN]
        else:
            current = [ALL_TOKEN]

    st.session_state[widget_key] = current
    st.session_state[prev_key] = current



def _apply_filter_state() -> None:
    options_map = st.session_state.get('_filter_options_map', {})
    filters = {}
    for col, _ in FILTER_FIELDS:
        valid_options = list(options_map.get(col, []))
        widget_value = st.session_state.get(f'filter_{col}', [ALL_TOKEN])
        filters[col] = _widget_to_filter_values(widget_value, valid_options)
    st.session_state['global_filters'] = filters



def _reset_filter_state() -> None:
    options_map = st.session_state.get('_filter_options_map', {})
    filters = {}
    for col, _ in FILTER_FIELDS:
        widget_key = f'filter_{col}'
        prev_key = _widget_prev_key(col)
        st.session_state[widget_key] = [ALL_TOKEN]
        st.session_state[prev_key] = [ALL_TOKEN]
        filters[col] = list(options_map.get(col, []))
    st.session_state['global_filters'] = filters



def init_filter_state(base_df: pd.DataFrame) -> None:
    options_map = {col: _get_options(base_df, col) for col, _ in FILTER_FIELDS}
    st.session_state['_filter_options_map'] = options_map

    if 'global_filters' not in st.session_state:
        st.session_state['global_filters'] = {col: list(options_map[col]) for col, _ in FILTER_FIELDS}

    current = dict(st.session_state['global_filters'])
    for col, _ in FILTER_FIELDS:
        valid_options = list(options_map[col])
        widget_key = f'filter_{col}'
        prev_key = _widget_prev_key(col)
        current_vals = _normalize_list(current.get(col, valid_options))
        current_vals = [v for v in current_vals if v in valid_options]

        if not current_vals or set(current_vals) == set(valid_options):
            desired_widget_vals = [ALL_TOKEN]
        else:
            desired_widget_vals = current_vals

        if widget_key not in st.session_state:
            st.session_state[widget_key] = desired_widget_vals
        if prev_key not in st.session_state:
            st.session_state[prev_key] = st.session_state[widget_key]

        current[col] = _widget_to_filter_values(st.session_state[widget_key], valid_options)

    st.session_state['global_filters'] = current



def render_sidebar_filters(base_df: pd.DataFrame) -> dict:
    init_filter_state(base_df)

    with st.sidebar:
        st.markdown("<div class='sidebar-title'>页面导航</div>", unsafe_allow_html=True)
        st.markdown('---')
        st.markdown("<div class='sidebar-title'>全局筛选</div>", unsafe_allow_html=True)
        st.markdown("<div class='sidebar-note'>默认选中‘全部’。若想多选具体项，请先取消‘全部’，再勾选需要的值。</div>", unsafe_allow_html=True)
        st.markdown("<div class='sidebar-note'>岗位类别和城市按拼音排序；学历、经验、公司规模、薪资单位按结构顺序排列。</div>", unsafe_allow_html=True)

        for col, label in FILTER_FIELDS:
            options = [ALL_TOKEN] + st.session_state['_filter_options_map'].get(col, [])
            st.multiselect(
                label,
                options,
                key=f'filter_{col}',
                placeholder=f'请选择{label}',
                on_change=_handle_multiselect_change,
                args=(col,),
            )

        c1, c2 = st.columns(2)
        with c1:
            st.button('应用筛选', use_container_width=True, on_click=_apply_filter_state)
        with c2:
            st.button('重置筛选', use_container_width=True, on_click=_reset_filter_state)

    return st.session_state['global_filters']



def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    out = df.copy()
    options_map = st.session_state.get('_filter_options_map', {})
    for col, values in filters.items():
        if col not in out.columns:
            continue
        selected = _normalize_list(values)
        all_options = list(options_map.get(col, []))
        if not selected or set(selected) == set(all_options):
            continue
        out = out[out[col].astype(str).str.strip().isin(selected)].copy()
    return out



def render_active_filters(filters: dict) -> None:
    options_map = st.session_state.get('_filter_options_map', {})
    active = []
    for col, label in FILTER_FIELDS:
        selected = _normalize_list(filters.get(col, []))
        all_options = list(options_map.get(col, []))
        if not selected or set(selected) == set(all_options):
            continue
        active.append(f"{label}：{', '.join(selected)}")
    if active:
        content = ' ｜ '.join(active)
    else:
        content = '全部样本'
    st.markdown(
        f"<div class='filter-chip-box'><div class='filter-chip-title'>当前筛选</div><div class='filter-chip-content'>{content}</div></div>",
        unsafe_allow_html=True,
    )
