# -*- coding: utf-8 -*-
from pathlib import Path
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt


# =========================
# 0. 路径与基础设置
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = BASE_DIR / "data" / "processed" / "clean_jobs_filtered.csv"
TABLE_DIR = BASE_DIR / "output" / "eda_tables"
FIGURE_DIR = BASE_DIR / "output" / "eda_figures"
SUMMARY_PATH = TABLE_DIR / "eda_summary.txt"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


# =========================
# 1. 通用工具函数
# =========================
def append_section(summary_lines, title, obj=None):
    """向 eda_summary.txt 追加一个分节。"""
    summary_lines.append(f"\n===== {title} =====")
    if obj is None:
        return

    if isinstance(obj, pd.DataFrame):
        summary_lines.append(obj.to_string())
    elif isinstance(obj, pd.Series):
        summary_lines.append(obj.to_string())
    else:
        summary_lines.append(str(obj))


def clean_text_series(series):
    s = series.dropna().astype(str).str.strip()
    return s[s != ""]


def save_dataframe(df, filename):
    df.to_csv(TABLE_DIR / filename, index=False, encoding="utf-8-sig")


def save_series_as_table(series, filename, index_name="category", value_name="count"):
    table_df = series.reset_index()
    table_df.columns = [index_name, value_name]
    save_dataframe(table_df, filename)
    return table_df


def save_bar_chart(
    series,
    fig_name,
    xlabel,
    ylabel,
    title=None,
    rotate=0,
    figsize=(10, 6),
    add_value_label=True
):
    s = series.copy()

    plt.figure(figsize=figsize)
    plt.bar(s.index.astype(str), s.values)
    if title is not None:
        plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=rotate, ha="right" if rotate else "center")

    if add_value_label and len(s) > 0:
        ymax = max(s.values)
        offset = ymax * 0.01 if ymax else 0.1
        for i, v in enumerate(s.values):
            plt.text(i, v + offset, str(v), ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / fig_name, dpi=300, bbox_inches="tight")
    plt.close()


def save_horizontal_bar(
    series,
    fig_name,
    xlabel,
    ylabel,
    title=None,
    add_pct=False,
    figsize=(10, 6)
):
    s = series.copy().sort_values(ascending=True)
    total = s.sum() if len(s) else 0

    plt.figure(figsize=figsize)
    plt.barh(s.index.astype(str), s.values)
    if title is not None:
        plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if len(s) > 0:
        x_offset = max(s.values) * 0.01 if max(s.values) else 0.1
        for i, v in enumerate(s.values):
            if add_pct and total:
                label = f"{v} ({v / total * 100:.1f}%)"
            else:
                label = str(v)
            plt.text(v + x_offset, i, label, va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / fig_name, dpi=300, bbox_inches="tight")
    plt.close()


def save_vertical_bar_with_pct(series, fig_name, xlabel, ylabel, title=None, figsize=(9, 6)):
    s = series.copy()
    total = s.sum() if len(s) else 0

    plt.figure(figsize=figsize)
    plt.bar(s.index.astype(str), s.values)
    if title is not None:
        plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    for i, v in enumerate(s.values):
        pct_text = f"\n{v / total * 100:.1f}%" if total else ""
        plt.text(i, v, f"{v}{pct_text}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / fig_name, dpi=300, bbox_inches="tight")
    plt.close()


def save_boxplot(
    df,
    group_col,
    value_col,
    order,
    fig_name,
    xlabel,
    ylabel,
    title=None,
    show_counts=False,
    figsize=(12, 6)
):
    plot_df = df[[group_col, value_col]].dropna().copy()
    order = [x for x in order if x in plot_df[group_col].unique()]
    if not order:
        return

    data = [plot_df.loc[plot_df[group_col] == g, value_col].values for g in order]
    labels = order.copy()

    if show_counts:
        counts = plot_df[group_col].value_counts()
        labels = [f"{g}\n(n={counts.get(g, 0)})" for g in order]

    plt.figure(figsize=figsize)
    plt.boxplot(data, labels=labels, patch_artist=False)
    if title is not None:
        plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / fig_name, dpi=300, bbox_inches="tight")
    plt.close()


def extract_skill_counter(series):
    counter = Counter()
    for value in series.dropna():
        items = [x.strip() for x in str(value).split("|")]
        for item in items:
            if item:
                counter[item] += 1
    return counter


def extract_skill_topn(series, top_n=20):
    counter = extract_skill_counter(series)
    return pd.DataFrame(counter.most_common(top_n), columns=["skill", "count"])


def build_skill_matrix_ratio(
    df,
    keyword_col="keyword",
    skill_col="skill_keywords_extract",
    top_keywords=8,
    top_skills=10
):
    """
    构建岗位 × 技能占比矩阵
    占比定义：
    某岗位中提及该技能的职位数 / 该岗位总职位数
    """
    top_keyword_list = df[keyword_col].value_counts().head(top_keywords).index.tolist()

    all_skill_counter = Counter()
    for value in df[skill_col].dropna():
        items = [x.strip() for x in str(value).split("|")]
        for item in items:
            if item:
                all_skill_counter[item] += 1

    top_skill_list = [x for x, _ in all_skill_counter.most_common(top_skills)]

    # 先做出现次数矩阵
    count_matrix = pd.DataFrame(0, index=top_keyword_list, columns=top_skill_list)

    for _, row in df[[keyword_col, skill_col]].dropna().iterrows():
        keyword = row[keyword_col]
        if keyword not in top_keyword_list:
            continue

        skills = [x.strip() for x in str(row[skill_col]).split("|")]
        for skill in skills:
            if skill in top_skill_list:
                count_matrix.loc[keyword, skill] += 1

    # 再转成岗位内占比
    keyword_total = df[df[keyword_col].isin(top_keyword_list)][keyword_col].value_counts()
    ratio_matrix = count_matrix.copy().astype(float)

    for kw in ratio_matrix.index:
        total_n = keyword_total.get(kw, 0)
        if total_n > 0:
            ratio_matrix.loc[kw, :] = ratio_matrix.loc[kw, :] / total_n
        else:
            ratio_matrix.loc[kw, :] = 0.0

    return ratio_matrix


def save_heatmap_ratio(matrix, fig_name, xlabel, ylabel, title=None, figsize=(11, 6)):
    plt.figure(figsize=figsize)
    data = matrix.values
    im = plt.imshow(data, aspect="auto")
    plt.colorbar(im)

    plt.xticks(range(len(matrix.columns)), matrix.columns, rotation=45, ha="right")
    plt.yticks(range(len(matrix.index)), matrix.index)

    if title is not None:
        plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            plt.text(j, i, f"{data[i, j]:.1%}", ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / fig_name, dpi=300, bbox_inches="tight")
    plt.close()


# =========================
# 2. 主程序
# =========================
def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"未找到数据文件：{INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)
    summary_lines = []

    print("===== EDA 开始 =====")
    print("数据文件：", INPUT_PATH)
    print("样本量：", len(df))

    # -------------------------------------------------
    # A. 基础概览
    # -------------------------------------------------
    append_section(summary_lines, "1. 基本信息", f"shape: {df.shape}\ncolumns: {df.columns.tolist()}")

    missing_df = pd.DataFrame({
        "column": df.columns,
        "missing_count": df.isna().sum().values,
        "missing_rate_pct": (df.isna().mean() * 100).round(2).values,
    }).sort_values("missing_rate_pct", ascending=False)
    append_section(summary_lines, "2. 缺失情况", missing_df)

    duplicate_df = pd.DataFrame({
        "metric": [
            "job_id重复数",
            "job_link重复数",
            "job_title+company_name重复数",
        ],
        "value": [
            int(df["job_id"].duplicated().sum()),
            int(df["job_link"].duplicated().sum()),
            int(df.duplicated(subset=["job_title", "company_name"]).sum()),
        ],
    })
    append_section(summary_lines, "3. 去重检查", duplicate_df)

    keyword_counts = df["keyword"].value_counts()
    city_counts_top20 = df["city"].value_counts().head(20)
    degree_counts_raw = df["degree_std"].value_counts(dropna=False)
    exp_counts_raw = df["experience_std"].value_counts(dropna=False)
    salary_unit_counts = df["salary_unit"].value_counts(dropna=False)

    append_section(summary_lines, "4. keyword分布", keyword_counts)
    append_section(summary_lines, "5. 城市分布 Top20", city_counts_top20)
    append_section(summary_lines, "6. 学历分布", degree_counts_raw)
    append_section(summary_lines, "7. 经验分布", exp_counts_raw)
    append_section(summary_lines, "8. 薪资单位分布", salary_unit_counts)

    salary_by_unit_df = (
        df.groupby("salary_unit", dropna=False)["salary_avg"]
        .agg(["count", "mean", "median", "min", "max"])
        .round(2)
        .reset_index()
    )
    append_section(summary_lines, "9. 按薪资单位统计 salary_avg", salary_by_unit_df)

    monthly_df = df[df["salary_unit"] == "月薪"].copy()

    monthly_degree_df = (
        monthly_df.groupby("degree_std")["salary_avg"]
        .agg(["count", "mean", "median"])
        .round(2)
        .sort_values("mean", ascending=False)
        .reset_index()
    )
    append_section(summary_lines, "10. 月薪按学历分组", monthly_degree_df)

    monthly_exp_df = (
        monthly_df.groupby("experience_std")["salary_avg"]
        .agg(["count", "mean", "median"])
        .round(2)
        .sort_values("mean", ascending=False)
        .reset_index()
    )
    append_section(summary_lines, "11. 月薪按经验分组", monthly_exp_df)

    monthly_keyword_df = (
        monthly_df.groupby("keyword")["salary_avg"]
        .agg(["count", "mean", "median"])
        .round(2)
        .sort_values("mean", ascending=False)
        .reset_index()
    )
    append_section(summary_lines, "12. 月薪按keyword分组", monthly_keyword_df)

    company_size_top20 = df["company_size_raw"].value_counts().head(20)
    company_type_top20 = df["company_type_raw"].value_counts().head(20)
    append_section(summary_lines, "13. 公司规模分布", company_size_top20)
    append_section(summary_lines, "14. 公司行业分布", company_type_top20)

    skill_top30_df = pd.DataFrame(
        extract_skill_counter(df["skill_keywords_extract"]).most_common(30),
        columns=["skill", "count"]
    )
    append_section(summary_lines, "15. 技能词频 Top30", skill_top30_df)

    # 新增：岗位×技能占比矩阵摘要
    skill_ratio_matrix = build_skill_matrix_ratio(
        df,
        keyword_col="keyword",
        skill_col="skill_keywords_extract",
        top_keywords=8,
        top_skills=10
    )
    append_section(summary_lines, "16. 岗位×技能占比矩阵（Top8岗位 × Top10技能）", skill_ratio_matrix.round(4))

    # -------------------------------------------------
    # B. 统计图表
    # -------------------------------------------------
    # 图1 keyword 分布
    save_horizontal_bar(
        series=keyword_counts,
        fig_name="keyword_distribution.png",
        xlabel="样本数",
        ylabel="岗位关键词",
        add_pct=True,
    )
    save_series_as_table(keyword_counts, "keyword_distribution.csv", "keyword", "count")

    # 图2 城市 Top15
    city_counts_top15 = df["city"].value_counts().head(15)
    save_horizontal_bar(
        series=city_counts_top15,
        fig_name="city_top15_distribution.png",
        xlabel="样本数",
        ylabel="城市",
        add_pct=False,
    )
    save_series_as_table(city_counts_top15, "city_top15_distribution.csv", "city", "count")

    # 图3 学历分布
    degree_order = ["不限", "大专", "本科", "硕士", "博士", "其他/未说明"]
    degree_counts = df["degree_std"].value_counts().reindex(degree_order, fill_value=0)
    degree_counts = degree_counts[degree_counts > 0]
    save_vertical_bar_with_pct(
        series=degree_counts,
        fig_name="degree_distribution.png",
        xlabel="学历要求",
        ylabel="样本数",
    )
    save_series_as_table(degree_counts, "degree_distribution.csv", "degree_std", "count")

    # 图4 经验分布
    exp_order = ["不限", "1年以内", "1-3年", "3-5年", "5-10年", "10年以上", "应届/在校", "其他/未说明"]
    exp_counts = df["experience_std"].value_counts().reindex(exp_order, fill_value=0)
    exp_counts = exp_counts[exp_counts > 0]
    save_vertical_bar_with_pct(
        series=exp_counts,
        fig_name="experience_distribution.png",
        xlabel="经验要求",
        ylabel="样本数",
    )
    save_series_as_table(exp_counts, "experience_distribution.csv", "experience_std", "count")

    # 图5 月薪按 keyword 箱线图（按中位数排序）
    keyword_salary_count = monthly_df["keyword"].value_counts()
    valid_keywords = keyword_salary_count[keyword_salary_count >= 20].index.tolist()
    monthly_df_plot = monthly_df[monthly_df["keyword"].isin(valid_keywords)].copy()

    keyword_salary_median = (
        monthly_df_plot.groupby("keyword")["salary_avg"]
        .median()
        .sort_values(ascending=False)
    )
    keyword_order_median = keyword_salary_median.index.tolist()

    save_boxplot(
        df=monthly_df_plot,
        group_col="keyword",
        value_col="salary_avg",
        order=keyword_order_median,
        fig_name="salary_boxplot_by_keyword.png",
        xlabel="岗位关键词",
        ylabel="月薪（元）",
        show_counts=True,
    )

    keyword_salary_median_df = keyword_salary_median.reset_index()
    keyword_salary_median_df.columns = ["keyword", "median_salary"]
    save_dataframe(keyword_salary_median_df, "salary_keyword_median_order.csv")

    # 图6 技能 Top20
    skill_top20_df = extract_skill_topn(df["skill_keywords_extract"], top_n=20)
    skill_series_top20 = pd.Series(skill_top20_df["count"].values, index=skill_top20_df["skill"].values)
    save_horizontal_bar(
        series=skill_series_top20,
        fig_name="skill_top20_distribution.png",
        xlabel="出现次数",
        ylabel="技能",
        add_pct=False,
    )
    save_dataframe(skill_top20_df, "skill_top20_distribution.csv")

    # 图7 平均月薪对比图
    salary_mean_df = (
        monthly_df.groupby("keyword")["salary_avg"]
        .agg(["count", "mean", "median"])
        .round(2)
        .sort_values("mean", ascending=False)
        .reset_index()
    )
    save_dataframe(salary_mean_df, "mean_salary_by_keyword.csv")

    salary_mean_series = pd.Series(salary_mean_df["mean"].values, index=salary_mean_df["keyword"].values)
    save_horizontal_bar(
        series=salary_mean_series,
        fig_name="mean_salary_by_keyword.png",
        xlabel="平均月薪（元）",
        ylabel="岗位关键词",
        add_pct=False,
    )

    # 图8 岗位关键词 × 核心技能占比热力图（标准化版）
    skill_ratio_matrix_reset = skill_ratio_matrix.reset_index().rename(columns={"index": "keyword"})
    save_dataframe(skill_ratio_matrix_reset, "keyword_skill_heatmap.csv")

    save_heatmap_ratio(
        matrix=skill_ratio_matrix,
        fig_name="keyword_skill_heatmap.png",
        xlabel="技能",
        ylabel="岗位关键词",
        title="岗位 × 核心技能占比热力图"
    )

    # 图9 经验要求 × 月薪箱线图
    exp_salary_order = ["不限", "1年以内", "1-3年", "3-5年", "5-10年", "10年以上"]
    exp_salary_order = [x for x in exp_salary_order if x in monthly_df["experience_std"].dropna().unique()]

    save_boxplot(
        df=monthly_df,
        group_col="experience_std",
        value_col="salary_avg",
        order=exp_salary_order,
        fig_name="salary_boxplot_by_experience.png",
        xlabel="经验要求",
        ylabel="月薪（元）",
        show_counts=True,
    )

    exp_salary_summary = (
        monthly_df.groupby("experience_std")["salary_avg"]
        .agg(["count", "mean", "median", "min", "max"])
        .round(2)
        .reset_index()
    )
    save_dataframe(exp_salary_summary, "salary_summary_by_experience.csv")

    # -------------------------------------------------
    # C. 保存摘要文件 + 控制台输出
    # -------------------------------------------------
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    print("\n===== EDA 已完成 =====")
    print("表格输出目录：", TABLE_DIR)
    print("图片输出目录：", FIGURE_DIR)
    print("摘要文件：", SUMMARY_PATH)
    print("\n前10条技能词频：")
    print(skill_top30_df.head(10))
    print("\n岗位×技能占比矩阵预览：")
    print(skill_ratio_matrix.round(4).head())


if __name__ == "__main__":
    main()