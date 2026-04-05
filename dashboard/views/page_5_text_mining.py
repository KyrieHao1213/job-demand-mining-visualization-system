# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import streamlit as st


CORE_KEYWORDS = [
    "数据分析师",
    "BI分析师",
    "用户分析师",
    "经营分析师",
    "商业分析师",
]

TOPIC_ORDER = [
    "用户需求与产品系统主题",
    "商业经营与策略洞察主题",
    "统计报表与业务支持主题",
    "数据建模与运营决策主题",
    "财务经营与预算管理主题",
]


# =========================
# 1. 基础工具函数
# =========================
def _get_base_dir(context: Dict[str, Any]) -> Path:
    """
    优先从 context 中取项目根目录；
    取不到时，按 dashboard/views/page_xxx.py -> 项目根目录 回推。
    """
    base_dir = context.get("base_dir")
    if base_dir is not None:
        return Path(base_dir)

    return Path(__file__).resolve().parents[2]


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _normalize_filter_keyword(filters: Dict[str, Any]) -> List[str]:
    """
    兼容你当前系统里可能出现的几种 keyword 筛选格式：
    - "全部"
    - ["全部"]
    - ["数据分析师", "商业分析师"]
    - None
    """
    raw = filters.get("keyword")

    if raw is None:
        return []

    if isinstance(raw, str):
        if raw == "全部":
            return []
        return [raw]

    if isinstance(raw, list):
        values = [str(x).strip() for x in raw if str(x).strip()]
        values = [x for x in values if x != "全部"]
        return values

    return []


def _build_bar(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    orientation: str = "h",
    color: Optional[str] = None,
    color_map: Optional[Dict[str, str]] = None,
    text_auto: bool = False,
    height: int = 360,
):
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        st.warning(f"{title}：数据为空或字段缺失。")
        return

    plot_df = df.copy()

    if orientation == "h":
        plot_df = plot_df.sort_values(x_col, ascending=True)
    else:
        plot_df = plot_df.sort_values(y_col, ascending=False)

    fig = px.bar(
        plot_df,
        x=x_col,
        y=y_col,
        orientation=orientation,
        color=color,
        color_discrete_map=color_map,
        text_auto=".1f" if text_auto else False,
    )
    fig.update_layout(
        title=title,
        template="plotly_white",
        margin=dict(l=20, r=20, t=60, b=20),
        height=height,
        showlegend=True if color else False,
    )
    st.plotly_chart(fig, use_container_width=True)


def _build_line(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    height: int = 360,
):
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        st.warning(f"{title}：数据为空或字段缺失。")
        return

    plot_df = df.sort_values(x_col).copy()
    fig = px.line(
        plot_df,
        x=x_col,
        y=y_col,
        markers=True,
        template="plotly_white",
    )
    fig.update_layout(
        title=title,
        margin=dict(l=20, r=20, t=60, b=20),
        height=height,
    )
    st.plotly_chart(fig, use_container_width=True)


# =========================
# 2. 岗位聚焦逻辑
# =========================
def _resolve_focus_keyword(
    filters: Dict[str, Any],
    available_keywords: List[str],
) -> Optional[str]:
    """
    文本页岗位聚焦规则：
    1. 若全局筛选只命中 1 个核心岗位，自动聚焦
    2. 若命中多个核心岗位，页内二次选择
    3. 若未命中核心岗位，则默认给一个可选框
    """
    selected_keywords = _normalize_filter_keyword(filters)
    selected_core = [x for x in selected_keywords if x in CORE_KEYWORDS and x in available_keywords]

    # 只命中一个核心岗位：自动聚焦
    if len(selected_core) == 1:
        st.caption(f"当前全局筛选已唯一命中核心岗位：**{selected_core[0]}**，页面自动聚焦到该岗位。")
        return selected_core[0]

    # 命中多个核心岗位：页内二次选择
    if len(selected_core) > 1:
        st.info("当前全局筛选包含多个核心岗位，因此这里保留页内二次选择。")
        return st.selectbox(
            "选择岗位查看关键词结果",
            options=selected_core,
            index=0,
            key="text_page_focus_keyword_multi"
        )

    # 全局筛选没有明确命中核心岗位：给全量核心岗位可选
    fallback_options = [x for x in CORE_KEYWORDS if x in available_keywords]
    if not fallback_options:
        return None

    return st.selectbox(
        "选择岗位查看关键词结果",
        options=fallback_options,
        index=0,
        key="text_page_focus_keyword_fallback"
    )


# =========================
# 3. 五主题概览卡片
# =========================
def _render_topic_cards(final_topics_df: pd.DataFrame):
    st.subheader("K=5 五主题概览")

    if final_topics_df.empty:
        st.warning("未检测到最终 K=5 主题词表。请先重新运行 lda_analysis.py。")
        return

    required_cols = {"topic_name", "rank", "term"}
    if not required_cols.issubset(final_topics_df.columns):
        st.warning("最终主题词表缺少必要字段。")
        return

    # 每个主题取前5个词
    topic_cards = []
    for topic_name in TOPIC_ORDER:
        sub = final_topics_df[final_topics_df["topic_name"] == topic_name].copy()
        if sub.empty:
            continue
        sub = sub.sort_values("rank").head(5)
        terms = "、".join(sub["term"].tolist())
        topic_cards.append((topic_name, terms))

    # 3 + 2 布局
    row1 = st.columns(3)
    row2 = st.columns(2)

    for idx, (topic_name, terms) in enumerate(topic_cards):
        target_col = row1[idx] if idx < 3 else row2[idx - 3]
        with target_col:
            st.markdown(
                f"""
                <div style="
                    background:#ffffff;
                    border:1px solid #dbe7f3;
                    border-radius:14px;
                    padding:16px 16px 12px 16px;
                    min-height:140px;
                    box-shadow:0 2px 8px rgba(0,0,0,0.04);
                ">
                    <div style="font-weight:700;font-size:18px;color:#1f3251;margin-bottom:10px;">
                        {topic_name}
                    </div>
                    <div style="font-size:14px;line-height:1.8;color:#3d4b63;">
                        {terms}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


# =========================
# 4. 岗位 × 主题对应关系图
# =========================
def _render_keyword_topic_relation_chart(base_dir: Path):
    lda_table_dir = base_dir / "output" / "lda_tables"
    mean_path = lda_table_dir / "lda_keyword_topic_mean_distribution_k5.csv"

    mean_df = _safe_read_csv(mean_path)
    if mean_df.empty:
        st.warning("未检测到岗位×主题平均概率表，请先重新运行增强版 lda_analysis.py。")
        return

    required_cols = {"keyword", "topic_name", "mean_topic_prob"}
    if not required_cols.issubset(mean_df.columns):
        st.warning("lda_keyword_topic_mean_distribution_k5.csv 缺少必要字段。")
        return

    pivot_df = mean_df.pivot(
        index="keyword",
        columns="topic_name",
        values="mean_topic_prob"
    )

    keyword_order = [x for x in CORE_KEYWORDS if x in pivot_df.index]
    pivot_df = pivot_df.reindex(keyword_order)

    topic_order = [x for x in TOPIC_ORDER if x in pivot_df.columns]
    pivot_df = pivot_df[topic_order]

    fig = px.imshow(
        pivot_df,
        text_auto=".1%",
        aspect="auto",
        color_continuous_scale="Blues"
    )
    fig.update_layout(
        title="岗位 × 主题平均概率热力图",
        xaxis_title="主题",
        yaxis_title="岗位类别",
        coloraxis_colorbar_title="平均概率",
        template="plotly_white",
        margin=dict(l=20, r=20, t=60, b=20),
        height=420,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption("说明：该图表示某一岗位文本在各主题上的平均概率，颜色越深，说明该岗位整体上越偏向对应主题。")


# =========================
# 5. 页面主函数
# =========================
def render(context: Dict[str, Any]):
    base_dir = _get_base_dir(context)
    filters = context.get("filters", {})

    keyword_table_dir = base_dir / "output" / "keyword_tables"
    lda_table_dir = base_dir / "output" / "lda_tables"

    tfidf_global_path = keyword_table_dir / "tfidf_global_filtered_top30.csv"
    textrank_global_path = keyword_table_dir / "textrank_global_filtered_top30.csv"
    tfidf_by_keyword_path = keyword_table_dir / "tfidf_by_keyword_top20.csv"
    textrank_by_keyword_path = keyword_table_dir / "textrank_by_keyword_top20.csv"

    lda_perplexity_path = lda_table_dir / "lda_perplexity_by_k.csv"
    lda_final_topics_path = lda_table_dir / "lda_final_topics_k5.csv"

    tfidf_global = _safe_read_csv(tfidf_global_path)
    textrank_global = _safe_read_csv(textrank_global_path)
    tfidf_by_keyword = _safe_read_csv(tfidf_by_keyword_path)
    textrank_by_keyword = _safe_read_csv(textrank_by_keyword_path)
    lda_perplexity = _safe_read_csv(lda_perplexity_path)
    lda_final_topics = _safe_read_csv(lda_final_topics_path)

    st.header("文本挖掘页")
    st.write("将 TF-IDF、TextRank 与 LDA 主题模型放在同一页中展示，突出你这个项目的 NLP 特色。")

    # =========================
    # 岗位聚焦对象
    # =========================
    available_keywords = []
    if not tfidf_by_keyword.empty and "keyword" in tfidf_by_keyword.columns:
        available_keywords = sorted(tfidf_by_keyword["keyword"].dropna().astype(str).unique().tolist())

    focus_keyword = _resolve_focus_keyword(filters, available_keywords)
    if focus_keyword is None:
        st.warning("当前没有可用的核心岗位关键词结果。")
        return

    # =========================
    # 全样本 TF-IDF / TextRank
    # =========================
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("全样本 TF-IDF Top10")
        if not tfidf_global.empty and {"term", "mean_tfidf"}.issubset(tfidf_global.columns):
            sub = tfidf_global.head(10).copy()
            _build_bar(
                sub,
                x_col="mean_tfidf",
                y_col="term",
                title="全样本 TF-IDF Top10",
                orientation="h",
                height=340
            )
        else:
            st.warning("未检测到全样本 TF-IDF 结果。")

    with col2:
        st.subheader("全样本 TextRank Top10")
        if not textrank_global.empty and {"term", "doc_freq"}.issubset(textrank_global.columns):
            sub = textrank_global.head(10).copy()
            _build_bar(
                sub,
                x_col="doc_freq",
                y_col="term",
                title="全样本 TextRank Top10",
                orientation="h",
                height=340
            )
        else:
            st.warning("未检测到全样本 TextRank 结果。")

    # =========================
    # 岗位内 TF-IDF / TextRank
    # =========================
    col3, col4 = st.columns(2)

    with col3:
        st.subheader(f"{focus_keyword} TF-IDF Top10")
        if not tfidf_by_keyword.empty and {"keyword", "version", "term", "mean_tfidf"}.issubset(tfidf_by_keyword.columns):
            sub = tfidf_by_keyword[
                (tfidf_by_keyword["keyword"] == focus_keyword) &
                (tfidf_by_keyword["version"] == "filtered")
            ].copy()

            if not sub.empty:
                sub = sub.sort_values("rank").head(10)
                _build_bar(
                    sub,
                    x_col="mean_tfidf",
                    y_col="term",
                    title=f"{focus_keyword} TF-IDF Top10",
                    orientation="h",
                    height=340
                )
            else:
                st.warning(f"未检测到 {focus_keyword} 的 TF-IDF 结果。")
        else:
            st.warning("TF-IDF 分岗位结果缺失。")

    with col4:
        st.subheader(f"{focus_keyword} TextRank Top10")
        if not textrank_by_keyword.empty and {"keyword", "version", "term", "doc_freq"}.issubset(textrank_by_keyword.columns):
            sub = textrank_by_keyword[
                (textrank_by_keyword["keyword"] == focus_keyword) &
                (textrank_by_keyword["version"] == "filtered")
            ].copy()

            if not sub.empty:
                sub = sub.sort_values("rank").head(10)
                _build_bar(
                    sub,
                    x_col="doc_freq",
                    y_col="term",
                    title=f"{focus_keyword} TextRank Top10",
                    orientation="h",
                    height=340
                )
            else:
                st.warning(f"未检测到 {focus_keyword} 的 TextRank 结果。")
        else:
            st.warning("TextRank 分岗位结果缺失。")

    # =========================
    # LDA 主题模型
    # =========================
    st.subheader("LDA 主题模型")
    st.write("候选主题数 K 的困惑度曲线如下，当前系统与论文保持一致，采用 K=5 作为最终主题数。")

    if not lda_perplexity.empty and {"k", "perplexity"}.issubset(lda_perplexity.columns):
        _build_line(
            lda_perplexity,
            x_col="k",
            y_col="perplexity",
            title="候选主题数 K 的困惑度曲线",
            height=360
        )
    else:
        st.warning("未检测到 LDA 困惑度结果。")

    _render_topic_cards(lda_final_topics)

    st.markdown("---")
    _render_keyword_topic_relation_chart(base_dir)

    st.success("答辩建议：这一页最能体现 NLP 项目特色。建议按“关键词层—语义中心层—潜在主题层”三步来讲，并与报告 4.3—4.4 保持一致。")