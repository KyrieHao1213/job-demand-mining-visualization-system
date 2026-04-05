# -*- coding: utf-8 -*-
"""
high_salary_keyword_diff.py
用于数据分析岗位招聘文本的高薪组 vs 非高薪组 TF-IDF 差异分析（第一轮：标准结果版）

当前版本功能：
1. 基于全部有效月薪样本，使用 75 分位数定义高薪岗位
2. 聚焦五类核心岗位样本做文本差异分析
3. 文本输入使用 job_title + job_desc_raw
4. 主体沿用 keyword_analysis.py 的预处理主线
5. 对停用词做“高薪差异版轻量调整”
6. 采用统一词表，在共享 TF-IDF 空间下比较：
   - 整体高薪组 vs 非高薪组
   - 数据分析师内部高薪组 vs 非高薪组
   - 商业分析师内部高薪组 vs 非高薪组
7. 输出差异词表（高薪更突出 Top15 + 非高薪更突出 Top15）
8. 输出 3 张差异条形图
9. 生成 high_salary_keyword_diff_summary.txt
"""

from pathlib import Path
import re
import jieba
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


# =========================
# 1. 路径配置
# =========================
BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_PATH = BASE_DIR / "data" / "processed" / "clean_jobs_filtered.csv"

OUTPUT_DIR = BASE_DIR / "output"
TABLE_DIR = OUTPUT_DIR / "high_salary_keyword_tables"
FIGURE_DIR = OUTPUT_DIR / "high_salary_keyword_figures"
SUMMARY_PATH = TABLE_DIR / "high_salary_keyword_diff_summary.txt"

STOPWORDS_PATH = BASE_DIR / "data" / "processed" / "stopwords.txt"


# =========================
# 2. 研究对象配置
# =========================
HIGH_SALARY_QUANTILE = 0.75
TOP_N = 15

CORE_KEYWORDS = [
    "数据分析师",
    "BI分析师",
    "用户分析师",
    "经营分析师",
    "商业分析师"
]

TARGET_KEYWORDS = [
    "数据分析师",
    "商业分析师"
]

KEYWORD_STUB_MAP = {
    "数据分析师": "data_analyst",
    "商业分析师": "business_analyst"
}


# =========================
# 3. 轻量保护：标准化映射
#    （主体复用 keyword_analysis.py）
# =========================
NORMALIZE_MAP = {
    "Power BI": "powerbi",
    "PowerBI": "powerbi",
    "power bi": "powerbi",
    "SQL Server": "sqlserver",
    "sql server": "sqlserver",
    "PostgreSQL": "postgresql",
    "postgresql": "postgresql",
    "MySQL": "mysql",
    "MYSQL": "mysql",
    "MongoDB": "mongodb",
    "mongoDB": "mongodb",
    "ElasticSearch": "elasticsearch",
    "Elasticsearch": "elasticsearch",
    "FineReport": "finereport",
    "FineBI": "finebi",
    "scikit-learn": "scikitlearn",
    "Scikit-learn": "scikitlearn",
    "A/B测试": "ab测试",
    "AB测试": "ab测试",
    "Python": "python",
    "PYTHON": "python",
    "Sql": "sql",
    "SQL": "sql",
    "Excel": "excel",
    "EXCEL": "excel",
    "Tableau": "tableau",
    "TABLEAU": "tableau",
    "SPSS": "spss",
    "SAS": "sas",
    "Oracle": "oracle",
    "Hive": "hive",
    "Spark": "spark",
    "ETL": "etl",
    "ROI": "roi",
    "R语言": "r",
    "C端": "c端",
    "B端": "b端",
    "GPT": "gpt",
    "LLM": "llm",
}


# =========================
# 4. 保护词
#    （主体复用 keyword_analysis.py）
# =========================
PROTECTED_WORDS = [
    "python", "sql", "excel", "tableau", "powerbi",
    "sqlserver", "mysql", "postgresql", "oracle",
    "mongodb", "redis", "elasticsearch",
    "hive", "spark", "etl",
    "spss", "sas", "roi",
    "scikitlearn", "ab测试",
    "finebi", "finereport",
    "gpt", "llm",
    "c端", "b端",
    "机器学习", "数据建模", "数据可视化"
]


# =========================
# 5. 招聘场景停用词
#    （主体复用 keyword_analysis.py）
# =========================
CUSTOM_STOPWORDS = {
    # 招聘动作词
    "负责", "参与", "协助", "跟进", "完成", "开展", "执行", "推进", "支持", "配合",
    # 要求描述词
    "熟悉", "了解", "掌握", "具有", "具备", "能够", "优先", "良好", "较强", "相关",
    "熟练", "熟练掌握", "优先考虑",
    # 岗位说明词
    "岗位", "职位", "工作", "任职", "要求", "公司", "部门", "团队",
    "岗位职责", "职位描述", "岗位描述", "任职要求", "任职条件", "职位要求",
    "岗位要求", "岗位内容", "职位信息", "职位详情", "职责描述",
    # 泛化连接词
    "以及", "并且", "等等", "相关工作", "人员", "日常", "以上",
    # 招聘文案高频噪音
    "我们", "提供", "使用", "工具", "描述", "条件", "欢迎", "加入",
    "候选人", "优先录用", "办公", "协同", "沟通", "表达", "学习",
    "毕业", "学历", "专业", "本科", "大专", "硕士", "博士",
    # 常见无效泛词
    "方面", "能力", "经验", "一定", "相关经验", "工作经验", "年以上",
    "即可", "优先者", "优先考虑", "优先录取"
}

CUSTOM_STOPWORDS.update({
    "进行", "通过", "包括", "需要", "其中", "实际", "基本", "优秀",
    "福利", "待遇", "方向", "结合", "针对", "面向", "例如",
    "以及相关", "相关内容", "相关事务", "相关事宜"
})

CUSTOM_STOPWORDS.update({
    "五险", "一金", "五险一金", "双休", "周末双休", "周末", "大小周",
    "年终奖", "奖金", "绩效", "提成",
    "节日福利", "福利", "补贴", "餐补", "房补", "交通补贴", "通讯补贴",
    "带薪", "年假", "带薪年假", "体检", "定期体检",
    "下午茶", "旅游", "团建", "培训", "晋升", "发展空间"
})

CUSTOM_STOPWORDS.update({
    "以上学历", "专业本科", "录用", "使用者",
    "其中", "实际", "基本", "例如", "比如", "相关方面"
})


# =========================
# 6. 高薪差异版：停用词轻量回调
# =========================
HIGH_SALARY_RESTORE_TERMS = {
    "决策", "优化", "策略", "模型", "指标",
    "业务", "经营", "财务", "用户", "产品"
}


# =========================
# 7. 结果层泛词过滤
#    仍然沿用 4.3 思路，但保留高薪差异版回调词
# =========================
GENERIC_TERMS_FILTER = {
    "数据", "分析", "数据分析", "分析师", "业务",
    "项目", "需求", "问题", "管理", "技术",
    "开发", "设计", "推动"
}

DIFF_GENERIC_FILTER = GENERIC_TERMS_FILTER - HIGH_SALARY_RESTORE_TERMS


# =========================
# 8. 通用工具函数
# =========================
def ensure_dirs():
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def save_table(df, file_name):
    save_path = TABLE_DIR / file_name
    df.to_csv(save_path, index=False, encoding="utf-8-sig")


def write_summary(lines):
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(str(line) + "\n")


def load_data():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"未找到输入文件: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)

    required_cols = ["salary_unit", "salary_avg", "keyword", "job_title", "job_desc_raw"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"缺少必要字段: {missing_cols}")

    return df


# =========================
# 9. 样本筛选与高薪划分
# =========================
def filter_valid_monthly_salary(df):
    out = df.copy()
    out = out[out["salary_unit"] == "月薪"].copy()
    out = out[out["salary_avg"].notna()].copy()
    return out


def compute_high_salary_threshold(df, quantile=HIGH_SALARY_QUANTILE):
    if df.empty:
        raise ValueError("有效月薪样本为空，无法计算高薪阈值。")
    return float(df["salary_avg"].quantile(quantile))


def label_high_salary_group(df, threshold):
    out = df.copy()
    out["is_high_salary"] = (out["salary_avg"] >= threshold).astype(int)
    out["salary_group"] = out["is_high_salary"].map({1: "高薪岗位", 0: "非高薪岗位"})
    return out


def filter_core_keyword_sample(df, core_keywords=None):
    if core_keywords is None:
        core_keywords = CORE_KEYWORDS

    out = df.copy()
    out = out[out["keyword"].isin(core_keywords)].copy()
    return out


def build_text_column(df):
    out = df.copy()

    out["job_title"] = out["job_title"].fillna("").astype(str)
    out["job_desc_raw"] = out["job_desc_raw"].fillna("").astype(str)

    out["text"] = (out["job_title"] + " " + out["job_desc_raw"]).str.replace(r"\s+", " ", regex=True).str.strip()
    out["text_norm"] = out["text"].apply(normalize_text)

    return out


# =========================
# 10. 复用 4.3 的预处理函数
# =========================
def register_protected_words():
    for word in PROTECTED_WORDS:
        jieba.add_word(word, freq=100000)


def load_stopwords(stopwords_path=STOPWORDS_PATH):
    stopwords = set()

    if stopwords_path.exists():
        with open(stopwords_path, "r", encoding="utf-8") as f:
            for line in f:
                word = line.strip()
                if word:
                    stopwords.add(word)

    stopwords.update(CUSTOM_STOPWORDS)

    # 高薪差异版轻量回调
    stopwords = stopwords - HIGH_SALARY_RESTORE_TERMS

    return stopwords


def normalize_text(text):
    if pd.isna(text):
        return ""

    text = str(text)

    for old, new in sorted(NORMALIZE_MAP.items(), key=lambda x: len(x[0]), reverse=True):
        text = text.replace(old, new)

    text = text.replace("ｒ", "r").replace("Ｒ", "r")
    text = text.replace("ａ", "a").replace("Ａ", "a")
    text = text.replace("ｉ", "i").replace("Ｉ", "i")

    text = re.sub(r"[①②③④⑤⑥⑦⑧⑨⑩]", " ", text)
    text = re.sub(r"\b[一二三四五六七八九十]+\b", " ", text)
    text = re.sub(r"\b\d+[、.)）]\s*", " ", text)

    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_meaningful_token(token, stopwords):
    if not token:
        return False

    token = token.strip()
    if not token:
        return False

    if token in stopwords:
        return False

    if re.fullmatch(r"\d+", token):
        return False

    if re.fullmatch(r"[_\W]+", token):
        return False

    if len(token) == 1 and re.fullmatch(r"[\u4e00-\u9fff]", token):
        return False

    return True


def tokenize_text(text, stopwords):
    text = normalize_text(text)
    words = jieba.lcut(text)

    clean_words = []
    for w in words:
        w = w.strip().lower()
        if is_meaningful_token(w, stopwords):
            clean_words.append(w)

    return clean_words


def tokenize_corpus(df, stopwords):
    out = df.copy()
    out["tokens"] = out["text"].apply(lambda x: tokenize_text(x, stopwords))
    out["text_cut"] = out["tokens"].apply(lambda x: " ".join(x))
    return out


def save_preprocess_preview(df, n=50):
    preview_cols = ["keyword", "salary_group", "job_title", "text", "tokens", "text_cut"]
    preview_df = df[preview_cols].head(n).copy()
    save_table(preview_df, "high_salary_keyword_diff_preview.csv")


# =========================
# 11. TF-IDF 差异分析函数
# =========================
def build_shared_tfidf(df_sub):
    work_df = df_sub.copy()
    work_df["text_cut"] = work_df["text_cut"].fillna("").astype(str)
    work_df = work_df[work_df["text_cut"].str.strip() != ""].copy()

    if work_df.empty:
        raise ValueError("text_cut 全为空，无法构造共享 TF-IDF。")

    vectorizer = TfidfVectorizer(
        tokenizer=str.split,
        preprocessor=None,
        token_pattern=None,
        lowercase=False,
        ngram_range=(1, 1)
    )

    tfidf_matrix = vectorizer.fit_transform(work_df["text_cut"])
    return work_df, vectorizer, tfidf_matrix


def compute_group_tfidf_diff(df_sub, top_n=TOP_N, excluded_terms=None):
    if excluded_terms is None:
        excluded_terms = set()

    work_df, vectorizer, tfidf_matrix = build_shared_tfidf(df_sub)

    if work_df["salary_group"].nunique() < 2:
        raise ValueError("当前比较样本中不足两类 salary_group，无法计算差异词表。")

    feature_names = vectorizer.get_feature_names_out()

    high_mask = (work_df["salary_group"] == "高薪岗位").values
    nonhigh_mask = (work_df["salary_group"] == "非高薪岗位").values

    if high_mask.sum() == 0 or nonhigh_mask.sum() == 0:
        raise ValueError("高薪组或非高薪组样本为空，无法计算差异词表。")

    high_matrix = tfidf_matrix[high_mask]
    nonhigh_matrix = tfidf_matrix[nonhigh_mask]

    mean_tfidf_high = high_matrix.mean(axis=0).A1
    mean_tfidf_nonhigh = nonhigh_matrix.mean(axis=0).A1

    doc_freq_high = (high_matrix > 0).sum(axis=0).A1
    doc_freq_nonhigh = (nonhigh_matrix > 0).sum(axis=0).A1

    diff_df = pd.DataFrame({
        "term": feature_names,
        "mean_tfidf_high": mean_tfidf_high,
        "mean_tfidf_nonhigh": mean_tfidf_nonhigh,
        "diff": mean_tfidf_high - mean_tfidf_nonhigh,
        "doc_freq_high": doc_freq_high.astype(int),
        "doc_freq_nonhigh": doc_freq_nonhigh.astype(int),
    })

    # 结果层过滤泛词
    diff_df = diff_df[~diff_df["term"].isin(excluded_terms)].copy()

    # 去掉两边都非常弱的词
    diff_df = diff_df[
        ~(
            (diff_df["mean_tfidf_high"] <= 0) &
            (diff_df["mean_tfidf_nonhigh"] <= 0)
        )
    ].copy()

    # 高薪更突出 TopN
    high_top = (
        diff_df[diff_df["diff"] > 0]
        .sort_values(
            by=["diff", "doc_freq_high", "mean_tfidf_high"],
            ascending=[False, False, False]
        )
        .head(top_n)
        .copy()
    )
    high_top["side"] = "高薪更突出"
    high_top["rank_side"] = range(1, len(high_top) + 1)

    # 非高薪更突出 TopN
    nonhigh_top = (
        diff_df[diff_df["diff"] < 0]
        .sort_values(
            by=["diff", "doc_freq_nonhigh", "mean_tfidf_nonhigh"],
            ascending=[True, False, False]
        )
        .head(top_n)
        .copy()
    )
    nonhigh_top["side"] = "非高薪更突出"
    nonhigh_top["rank_side"] = range(1, len(nonhigh_top) + 1)

    top_df = pd.concat([high_top, nonhigh_top], ignore_index=True)
    top_df = top_df[[
        "side", "rank_side", "term",
        "mean_tfidf_high", "mean_tfidf_nonhigh", "diff",
        "doc_freq_high", "doc_freq_nonhigh"
    ]].copy()

    return diff_df, top_df, work_df


def build_overall_diff_table(df, top_n=TOP_N):
    return compute_group_tfidf_diff(df, top_n=top_n, excluded_terms=DIFF_GENERIC_FILTER)


def build_keyword_inner_diff_table(df, keyword_name, top_n=TOP_N):
    sub_df = df[df["keyword"] == keyword_name].copy()
    return compute_group_tfidf_diff(sub_df, top_n=top_n, excluded_terms=DIFF_GENERIC_FILTER)


# =========================
# 12. 绘图函数
# =========================
def save_diff_bar_chart(diff_top_df, fig_name, title=None):
    plot_df = diff_top_df.copy()

    # 让“非高薪更突出”的词显示为负值，方便一张图里对比
    plot_df["plot_value"] = plot_df.apply(
        lambda row: row["diff"] if row["side"] == "高薪更突出" else row["diff"],
        axis=1
    )

    # 为了让图更有层次，按差值从小到大排序
    plot_df = plot_df.sort_values("plot_value", ascending=True)

    plt.figure(figsize=(10, 8))
    plt.barh(plot_df["term"], plot_df["plot_value"])
    if title is not None:
        plt.title(title)
    plt.xlabel("TF-IDF 差值（高薪组 - 非高薪组）")
    plt.ylabel("词项")

    for i, v in enumerate(plot_df["plot_value"]):
        plt.text(v, i, f"{v:.4f}", va="center", ha="left" if v >= 0 else "right", fontsize=8)

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / fig_name, dpi=300, bbox_inches="tight")
    plt.close()


# =========================
# 13. 摘要辅助函数
# =========================
def append_diff_preview(summary_lines, title, diff_top_df, n=10):
    summary_lines.append("")
    summary_lines.append(title)

    high_df = diff_top_df[diff_top_df["side"] == "高薪更突出"].head(n)
    nonhigh_df = diff_top_df[diff_top_df["side"] == "非高薪更突出"].head(n)

    summary_lines.append("[高薪更突出]")
    if high_df.empty:
        summary_lines.append("- 无")
    else:
        for _, row in high_df.iterrows():
            summary_lines.append(
                f"- {int(row['rank_side'])}. {row['term']} : diff={row['diff']:.6f}, "
                f"high={row['mean_tfidf_high']:.6f}, nonhigh={row['mean_tfidf_nonhigh']:.6f}, "
                f"doc_freq_high={int(row['doc_freq_high'])}, doc_freq_nonhigh={int(row['doc_freq_nonhigh'])}"
            )

    summary_lines.append("[非高薪更突出]")
    if nonhigh_df.empty:
        summary_lines.append("- 无")
    else:
        for _, row in nonhigh_df.iterrows():
            summary_lines.append(
                f"- {int(row['rank_side'])}. {row['term']} : diff={row['diff']:.6f}, "
                f"high={row['mean_tfidf_high']:.6f}, nonhigh={row['mean_tfidf_nonhigh']:.6f}, "
                f"doc_freq_high={int(row['doc_freq_high'])}, doc_freq_nonhigh={int(row['doc_freq_nonhigh'])}"
            )


# =========================
# 14. 主流程
# =========================
def main():
    ensure_dirs()
    register_protected_words()

    # 1) 读取数据
    df = load_data()
    n_all_filtered = len(df)

    # 2) 保留有效月薪样本，并基于全部月薪样本定义高薪阈值
    df_monthly = filter_valid_monthly_salary(df)
    n_monthly = len(df_monthly)

    threshold = compute_high_salary_threshold(df_monthly, quantile=HIGH_SALARY_QUANTILE)
    df_monthly = label_high_salary_group(df_monthly, threshold)

    # 3) 聚焦五类核心岗位
    df_core = filter_core_keyword_sample(df_monthly, CORE_KEYWORDS)
    n_core = len(df_core)

    # 4) 构造文本并分词
    df_core = build_text_column(df_core)
    stopwords = load_stopwords()
    df_core = tokenize_corpus(df_core, stopwords)

    # 5) 保存预处理预览
    save_preprocess_preview(df_core, n=50)

    # 6) 样本概况
    overall_group_count_df = (
        df_core["salary_group"]
        .value_counts()
        .rename_axis("salary_group")
        .reset_index(name="count")
    )

    target_keyword_count_df = (
        df_core.groupby(["keyword", "salary_group"], dropna=False)
        .size()
        .reset_index(name="count")
    )

    # 7) 整体高薪 vs 非高薪差异
    overall_diff_full_df, overall_diff_top_df, overall_work_df = build_overall_diff_table(df_core, top_n=TOP_N)
    save_table(overall_diff_top_df, "overall_tfidf_diff_top15.csv")

    # 8) 数据分析师内部差异
    data_diff_full_df, data_diff_top_df, data_work_df = build_keyword_inner_diff_table(df_core, "数据分析师", top_n=TOP_N)
    save_table(data_diff_top_df, "data_analyst_tfidf_diff_top15.csv")

    # 9) 商业分析师内部差异
    biz_diff_full_df, biz_diff_top_df, biz_work_df = build_keyword_inner_diff_table(df_core, "商业分析师", top_n=TOP_N)
    save_table(biz_diff_top_df, "business_analyst_tfidf_diff_top15.csv")

    # 10) 绘图
    save_diff_bar_chart(
        overall_diff_top_df,
        fig_name="overall_tfidf_diff_top15.png",
        title="整体高薪组 vs 非高薪组 TF-IDF 差异词（Top15×2）"
    )

    save_diff_bar_chart(
        data_diff_top_df,
        fig_name="data_analyst_tfidf_diff_top15.png",
        title="数据分析师内部高薪 vs 非高薪 TF-IDF 差异词（Top15×2）"
    )

    save_diff_bar_chart(
        biz_diff_top_df,
        fig_name="business_analyst_tfidf_diff_top15.png",
        title="商业分析师内部高薪 vs 非高薪 TF-IDF 差异词（Top15×2）"
    )

    # 11) 写 summary
    summary_lines = [
        "===== 高薪岗位关键词差异分析模块：第一轮（标准结果版） =====",
        f"全部过滤后样本量: {n_all_filtered}",
        f"有效月薪样本量: {n_monthly}",
        f"高薪阈值（全部有效月薪样本 75 分位数）: {threshold:.2f}",
        f"五类核心岗位有效月薪样本量: {n_core}",
        "",
        "五类核心岗位中的高薪组 / 非高薪组样本量："
    ]

    for _, row in overall_group_count_df.iterrows():
        summary_lines.append(f"- {row['salary_group']}: {int(row['count'])}")

    summary_lines.extend([
        "",
        "重点岗位内部样本量："
    ])

    for kw in TARGET_KEYWORDS:
        sub = target_keyword_count_df[target_keyword_count_df["keyword"] == kw].copy()
        summary_lines.append(f"[{kw}]")
        for _, row in sub.iterrows():
            summary_lines.append(f"- {row['salary_group']}: {int(row['count'])}")
        if sub.empty:
            summary_lines.append("- 无样本")

    summary_lines.extend([
        "",
        "方法说明：",
        "- 文本输入使用 job_title + job_desc_raw",
        "- 预处理主体沿用 4.3 关键词分析逻辑",
        "- 停用词采用高薪差异版轻量调整，并回调以下词项：",
        f"- {sorted(HIGH_SALARY_RESTORE_TERMS)}",
        "- 在共享词表空间下分别计算高薪组与非高薪组的平均 TF-IDF",
        "- 以 diff = mean_tfidf_high - mean_tfidf_nonhigh 衡量差异强度",
        "- 输出高薪更突出 Top15 与非高薪更突出 Top15",
    ])

    append_diff_preview(summary_lines, "整体高薪组 vs 非高薪组差异词预览（Top10）", overall_diff_top_df, n=10)
    append_diff_preview(summary_lines, "数据分析师内部差异词预览（Top10）", data_diff_top_df, n=10)
    append_diff_preview(summary_lines, "商业分析师内部差异词预览（Top10）", biz_diff_top_df, n=10)

    summary_lines.extend([
        "",
        "当前阶段已完成：",
        "1. 基于全部有效月薪样本计算高薪阈值",
        "2. 在五类核心岗位范围内划分高薪组与非高薪组",
        "3. 构造统一词表下的整体 TF-IDF 差异词表",
        "4. 构造数据分析师内部 TF-IDF 差异词表",
        "5. 构造商业分析师内部 TF-IDF 差异词表",
        "6. 输出差异词表与对应图形",
        "",
        "下一步建议：",
        "1. 补充高薪组 vs 非高薪组的 TextRank 差异分析",
        "2. 补充高薪组 vs 非高薪组的主题差异分析",
        "3. 将关键词差异结果与 4.5 的高薪岗位结构化特征分析进行综合解释",
    ])

    write_summary(summary_lines)

    # 12) 控制台输出
    print("高薪岗位关键词差异分析第一轮（标准结果版）运行完成。")
    print(f"全部过滤后样本量: {n_all_filtered}")
    print(f"有效月薪样本量: {n_monthly}")
    print(f"高薪阈值（75分位数）: {threshold:.2f}")
    print(f"五类核心岗位有效月薪样本量: {n_core}")
    print(f"预处理预览已保存到: {TABLE_DIR / 'high_salary_keyword_diff_preview.csv'}")
    print(f"整体差异词表已保存到: {TABLE_DIR / 'overall_tfidf_diff_top15.csv'}")
    print(f"数据分析师内部差异词表已保存到: {TABLE_DIR / 'data_analyst_tfidf_diff_top15.csv'}")
    print(f"商业分析师内部差异词表已保存到: {TABLE_DIR / 'business_analyst_tfidf_diff_top15.csv'}")
    print(f"摘要文件已保存到: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()