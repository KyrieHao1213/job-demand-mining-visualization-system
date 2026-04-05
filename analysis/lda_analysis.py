# -*- coding: utf-8 -*-
"""
lda_analysis.py
用于五类核心岗位招聘文本的 LDA 主题模型分析（增强版）

当前版本功能：
1. 使用五类核心岗位样本
2. 只使用 job_desc_raw 作为 LDA 输入语料
3. 做 LDA 轻量预处理
4. 使用 CountVectorizer 构造词频矩阵
5. 比较 K=4,5,6,7,8 的困惑度
6. 输出各 K 下主题关键词表
7. 输出最终 K=5 的主题词表
8. 输出每条文本的主题概率分布
9. 输出每类岗位的平均主题概率分布
10. 输出每类岗位的主导主题占比
11. 生成 topic_summary.txt
"""

from pathlib import Path
import re
import jieba
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


# =========================
# 1. 路径配置
# =========================
BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_PATH = BASE_DIR / "data" / "processed" / "clean_jobs_filtered.csv"

OUTPUT_DIR = BASE_DIR / "output"
TABLE_DIR = OUTPUT_DIR / "lda_tables"
FIGURE_DIR = OUTPUT_DIR / "lda_figures"
SUMMARY_PATH = TABLE_DIR / "topic_summary.txt"

STOPWORDS_PATH = BASE_DIR / "data" / "processed" / "stopwords.txt"


# =========================
# 2. 研究对象配置
# =========================
CORE_KEYWORDS = [
    "数据分析师",
    "BI分析师",
    "用户分析师",
    "经营分析师",
    "商业分析师"
]

K_CANDIDATES = [4, 5, 6, 7, 8]
FINAL_K = 5

# 注意：这里沿用你报告中的最终主题命名。
# 若你后续重新训练后发现 topic_id 对应顺序变化，可手工调整这个映射。
FINAL_TOPIC_NAME_MAP = {
    0: "用户需求与产品系统主题",
    1: "商业经营与策略洞察主题",
    2: "统计报表与业务支持主题",
    3: "数据建模与运营决策主题",
    4: "财务经营与预算管理主题",
}


# =========================
# 3. 英文工具词/复合表达标准化
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
# 5. LDA 轻量停用词
# =========================
CUSTOM_STOPWORDS = {
    "负责", "参与", "协助", "跟进", "完成", "开展", "执行", "推进", "支持", "配合",
    "熟悉", "了解", "掌握", "具有", "具备", "能够", "优先", "良好", "较强", "相关",
    "熟练", "熟练掌握", "优先考虑",
    "岗位", "职位", "工作", "任职", "要求", "公司", "部门", "团队",
    "岗位职责", "职位描述", "岗位描述", "任职要求", "任职条件", "职位要求",
    "岗位要求", "岗位内容", "职位信息", "职位详情", "职责描述",
    "以及", "并且", "等等", "相关工作", "人员", "日常", "以上", "and",
    "我们", "提供", "使用", "工具", "描述", "条件", "欢迎", "加入",
    "候选人", "优先录用", "办公", "协同", "沟通", "表达", "学习",
    "毕业", "学历", "专业", "本科", "大专", "硕士", "博士",
    "方面", "能力", "经验", "一定", "相关经验", "工作经验", "年以上",
    "即可", "优先者", "优先录取", "以上学历",
    "五险", "一金", "五险一金", "双休", "周末双休", "周末", "大小周",
    "年终奖", "奖金", "绩效", "提成",
    "节日福利", "福利", "补贴", "餐补", "房补", "交通补贴", "通讯补贴",
    "带薪", "年假", "带薪年假", "体检", "定期体检",
    "下午茶", "旅游", "团建", "培训", "晋升", "发展空间"
}


# =========================
# 6. 基础工具函数
# =========================
def ensure_dirs():
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def register_protected_words():
    for word in PROTECTED_WORDS:
        jieba.add_word(word, freq=100000)


def load_data():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"未找到输入文件: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)

    required_cols = ["keyword", "job_desc_raw"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"缺少必要字段: {missing_cols}")

    return df


def filter_core_keywords(df, core_keywords=None):
    if core_keywords is None:
        core_keywords = CORE_KEYWORDS
    out = df.copy()
    out = out[out["keyword"].isin(core_keywords)].copy()
    return out


def build_lda_text(df):
    out = df.copy()
    out["job_desc_raw"] = out["job_desc_raw"].fillna("").astype(str)
    out["lda_text"] = out["job_desc_raw"].str.replace(r"\s+", " ", regex=True).str.strip()
    return out


def load_stopwords(stopwords_path=STOPWORDS_PATH):
    stopwords = set()

    if stopwords_path.exists():
        with open(stopwords_path, "r", encoding="utf-8") as f:
            for line in f:
                word = line.strip()
                if word:
                    stopwords.add(word)

    stopwords.update(CUSTOM_STOPWORDS)
    return stopwords


def normalize_text(text):
    if pd.isna(text):
        return ""

    text = str(text)

    for old, new in sorted(NORMALIZE_MAP.items(), key=lambda x: len(x[0]), reverse=True):
        text = text.replace(old, new)

    text = re.sub(r"[①②③④⑤⑥⑦⑧⑨⑩]", " ", text)
    text = re.sub(r"\b[一二三四五六七八九十]+\b", " ", text)
    text = re.sub(r"\b\d+[、.)）]\s*", " ", text)

    fullwidth_map = {
        "ｒ": "r", "Ｒ": "r",
        "ａ": "a", "Ａ": "a",
        "ｉ": "i", "Ｉ": "i",
    }
    for old, new in fullwidth_map.items():
        text = text.replace(old, new)

    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_meaningful_token_for_lda(token, stopwords):
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


def tokenize_text_for_lda(text, stopwords):
    text = normalize_text(text)
    words = jieba.lcut(text)

    clean_words = []
    for w in words:
        w = w.strip().lower()
        if is_meaningful_token_for_lda(w, stopwords):
            clean_words.append(w)

    return clean_words


def tokenize_for_lda(df, stopwords):
    out = df.copy()
    out["lda_tokens"] = out["lda_text"].apply(lambda x: tokenize_text_for_lda(x, stopwords))
    out["lda_text_cut"] = out["lda_tokens"].apply(lambda x: " ".join(x))
    return out


def prepare_lda_corpus(df):
    """
    返回：
    - df_valid: 真正进入 vectorizer/lda 的有效文档 DataFrame
    - corpus:   对应的分词语料
    """
    out = df.copy()
    out["lda_text_cut"] = out["lda_text_cut"].fillna("").astype(str)
    out = out[out["lda_text_cut"].str.strip() != ""].copy()
    out = out.reset_index(drop=True)

    corpus = out["lda_text_cut"]
    return out, corpus


def save_table(df, file_name):
    save_path = TABLE_DIR / file_name
    df.to_csv(save_path, index=False, encoding="utf-8-sig")


def write_summary(lines):
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(str(line) + "\n")


# =========================
# 7. LDA 主题模型函数
# =========================
def fit_lda_candidates(
    corpus,
    k_candidates=None,
    max_features=2000,
    min_df=5,
    max_df=0.8,
    random_state=42
):
    if k_candidates is None:
        k_candidates = K_CANDIDATES

    vectorizer = CountVectorizer(
        tokenizer=str.split,
        preprocessor=None,
        token_pattern=None,
        lowercase=False,
        max_features=max_features,
        min_df=min_df,
        max_df=max_df
    )

    dtm = vectorizer.fit_transform(corpus)

    rows = []
    lda_models = {}

    for k in k_candidates:
        lda = LatentDirichletAllocation(
            n_components=k,
            random_state=random_state,
            learning_method="batch",
            max_iter=30
        )
        lda.fit(dtm)
        perplexity = lda.perplexity(dtm)

        rows.append({
            "k": k,
            "perplexity": perplexity,
            "n_features": dtm.shape[1],
            "n_docs": dtm.shape[0]
        })
        lda_models[k] = lda

    perplexity_df = pd.DataFrame(rows).sort_values("k").reset_index(drop=True)
    return vectorizer, dtm, perplexity_df, lda_models


def extract_topic_keywords(lda_models, vectorizer, top_n=15):
    feature_names = vectorizer.get_feature_names_out()
    rows = []

    for k, lda in lda_models.items():
        for topic_id, topic_weights in enumerate(lda.components_):
            top_idx = topic_weights.argsort()[::-1][:top_n]
            for rank, idx in enumerate(top_idx, start=1):
                rows.append({
                    "k": k,
                    "topic_id": topic_id,
                    "report_topic_id": topic_id + 1,
                    "topic_name": FINAL_TOPIC_NAME_MAP.get(topic_id, f"主题{topic_id + 1}") if k == FINAL_K else f"Topic {topic_id + 1}",
                    "rank": rank,
                    "term": feature_names[idx],
                    "weight": float(topic_weights[idx])
                })

    return pd.DataFrame(rows)


def extract_final_topic_keywords(lda_model, vectorizer, topic_name_map=None, top_n=15):
    if topic_name_map is None:
        topic_name_map = FINAL_TOPIC_NAME_MAP

    feature_names = vectorizer.get_feature_names_out()
    rows = []

    for topic_id, topic_weights in enumerate(lda_model.components_):
        top_idx = topic_weights.argsort()[::-1][:top_n]
        for rank, idx in enumerate(top_idx, start=1):
            rows.append({
                "topic_id": topic_id,
                "report_topic_id": topic_id + 1,
                "topic_name": topic_name_map.get(topic_id, f"主题{topic_id + 1}"),
                "rank": rank,
                "term": feature_names[idx],
                "weight": float(topic_weights[idx])
            })

    return pd.DataFrame(rows)


def build_doc_topic_distribution(df_valid, lda_model, dtm, topic_name_map=None):
    if topic_name_map is None:
        topic_name_map = FINAL_TOPIC_NAME_MAP

    topic_probs = lda_model.transform(dtm)
    n_topics = topic_probs.shape[1]

    out = df_valid[["keyword", "lda_text"]].copy()
    out["doc_id"] = range(len(out))

    for i in range(n_topics):
        out[f"topic_{i + 1}_prob"] = topic_probs[:, i]

    dominant_topic_idx = topic_probs.argmax(axis=1)
    dominant_topic_prob = topic_probs.max(axis=1)

    out["dominant_topic_id"] = dominant_topic_idx
    out["dominant_report_topic_id"] = out["dominant_topic_id"] + 1
    out["dominant_topic_name"] = out["dominant_topic_id"].map(
        lambda x: topic_name_map.get(x, f"主题{x + 1}")
    )
    out["dominant_topic_prob"] = dominant_topic_prob

    return out


def build_keyword_topic_mean_distribution(doc_topic_df, topic_name_map=None):
    if topic_name_map is None:
        topic_name_map = FINAL_TOPIC_NAME_MAP

    prob_cols = [col for col in doc_topic_df.columns if col.startswith("topic_") and col.endswith("_prob")]

    wide_df = (
        doc_topic_df.groupby("keyword")[prob_cols]
        .mean()
        .reset_index()
    )

    rows = []
    for _, row in wide_df.iterrows():
        keyword = row["keyword"]
        for i, col in enumerate(prob_cols):
            topic_id = i
            rows.append({
                "keyword": keyword,
                "topic_id": topic_id,
                "report_topic_id": topic_id + 1,
                "topic_name": topic_name_map.get(topic_id, f"主题{topic_id + 1}"),
                "mean_topic_prob": float(row[col])
            })

    return pd.DataFrame(rows)


def build_keyword_dominant_topic_share(doc_topic_df, topic_name_map=None):
    if topic_name_map is None:
        topic_name_map = FINAL_TOPIC_NAME_MAP

    ct = pd.crosstab(doc_topic_df["keyword"], doc_topic_df["dominant_topic_id"], normalize="index")
    ct = ct.fillna(0).reset_index()

    rows = []
    topic_cols = [col for col in ct.columns if col != "keyword"]

    for _, row in ct.iterrows():
        keyword = row["keyword"]
        for topic_id in topic_cols:
            rows.append({
                "keyword": keyword,
                "topic_id": int(topic_id),
                "report_topic_id": int(topic_id) + 1,
                "topic_name": topic_name_map.get(int(topic_id), f"主题{int(topic_id) + 1}"),
                "dominant_topic_share": float(row[topic_id])
            })

    return pd.DataFrame(rows)


# =========================
# 8. 主流程
# =========================
def main():
    ensure_dirs()
    register_protected_words()

    df = load_data()
    n_raw = len(df)

    df = filter_core_keywords(df, CORE_KEYWORDS)
    n_core = len(df)

    df = build_lda_text(df)
    stopwords = load_stopwords()
    df = tokenize_for_lda(df, stopwords)

    preview_cols = ["keyword", "lda_text", "lda_tokens", "lda_text_cut"]
    preview_df = df[preview_cols].head(30).copy()
    save_table(preview_df, "lda_preprocess_preview.csv")

    group_counts = (
        df["keyword"]
        .value_counts()
        .rename_axis("keyword")
        .reset_index(name="count")
    )
    save_table(group_counts, "lda_group_counts.csv")

    df_valid, corpus = prepare_lda_corpus(df)

    vectorizer, dtm, perplexity_df, lda_models = fit_lda_candidates(
        corpus=corpus,
        k_candidates=K_CANDIDATES,
        max_features=2000,
        min_df=5,
        max_df=0.8,
        random_state=42
    )
    save_table(perplexity_df, "lda_perplexity_by_k.csv")

    topics_df = extract_topic_keywords(
        lda_models=lda_models,
        vectorizer=vectorizer,
        top_n=15
    )
    save_table(topics_df, "lda_topics_by_k.csv")

    if FINAL_K not in lda_models:
        raise ValueError(f"未找到最终主题数 K={FINAL_K} 对应的模型。")

    final_lda = lda_models[FINAL_K]

    final_topics_df = extract_final_topic_keywords(
        lda_model=final_lda,
        vectorizer=vectorizer,
        topic_name_map=FINAL_TOPIC_NAME_MAP,
        top_n=15
    )
    save_table(final_topics_df, "lda_final_topics_k5.csv")

    doc_topic_df = build_doc_topic_distribution(
        df_valid=df_valid,
        lda_model=final_lda,
        dtm=dtm,
        topic_name_map=FINAL_TOPIC_NAME_MAP
    )
    save_table(doc_topic_df, "lda_doc_topic_distribution_k5.csv")

    keyword_topic_mean_df = build_keyword_topic_mean_distribution(
        doc_topic_df=doc_topic_df,
        topic_name_map=FINAL_TOPIC_NAME_MAP
    )
    save_table(keyword_topic_mean_df, "lda_keyword_topic_mean_distribution_k5.csv")

    keyword_dominant_share_df = build_keyword_dominant_topic_share(
        doc_topic_df=doc_topic_df,
        topic_name_map=FINAL_TOPIC_NAME_MAP
    )
    save_table(keyword_dominant_share_df, "lda_keyword_dominant_topic_share_k5.csv")

    summary_lines = [
        "===== 主题模型模块：LDA 增强版 =====",
        f"原始样本量: {n_raw}",
        f"进入五类核心岗位后的样本量: {n_core}",
        f"进入 LDA 的有效文档数: {len(corpus)}",
        f"停用词总数: {len(stopwords)}",
        f"最终主题数 FINAL_K: {FINAL_K}",
        "",
        "五类核心岗位样本分布："
    ]

    for _, row in group_counts.iterrows():
        summary_lines.append(f"- {row['keyword']}: {row['count']}")

    summary_lines.extend([
        "",
        "各 K 的困惑度："
    ])
    for _, row in perplexity_df.iterrows():
        summary_lines.append(f"- K={int(row['k'])}: perplexity={row['perplexity']:.4f}")

    summary_lines.extend([
        "",
        "最终 K=5 主题命名："
    ])
    for topic_id, topic_name in FINAL_TOPIC_NAME_MAP.items():
        summary_lines.append(f"- Topic {topic_id + 1}: {topic_name}")

    summary_lines.extend([
        "",
        "新增输出文件：",
        f"- {TABLE_DIR / 'lda_final_topics_k5.csv'}",
        f"- {TABLE_DIR / 'lda_doc_topic_distribution_k5.csv'}",
        f"- {TABLE_DIR / 'lda_keyword_topic_mean_distribution_k5.csv'}",
        f"- {TABLE_DIR / 'lda_keyword_dominant_topic_share_k5.csv'}",
        "",
        "当前阶段已完成：",
        "1. 五类核心岗位样本筛选",
        "2. LDA 语料构造（仅使用 job_desc_raw）",
        "3. LDA 轻量预处理与分词",
        "4. CountVectorizer 词频矩阵构建",
        "5. K=4,5,6,7,8 候选主题模型训练",
        "6. 各 K 困惑度输出",
        "7. 各 K 主题关键词表输出",
        "8. K=5 最终主题词表输出",
        "9. 每条文本主题概率分布输出",
        "10. 每类岗位平均主题概率分布输出",
        "11. 每类岗位主导主题占比输出",
    ])

    write_summary(summary_lines)

    print("LDA 增强版运行完成。")
    print(f"原始样本量: {n_raw}")
    print(f"五类核心岗位样本量: {n_core}")
    print(f"LDA 有效文档数: {len(corpus)}")
    print(f"最终主题数 K: {FINAL_K}")
    print(f"LDA 困惑度结果已保存到: {TABLE_DIR / 'lda_perplexity_by_k.csv'}")
    print(f"LDA 全部主题词结果已保存到: {TABLE_DIR / 'lda_topics_by_k.csv'}")
    print(f"LDA 最终主题词结果已保存到: {TABLE_DIR / 'lda_final_topics_k5.csv'}")
    print(f"LDA 文档主题分布已保存到: {TABLE_DIR / 'lda_doc_topic_distribution_k5.csv'}")
    print(f"LDA 岗位平均主题分布已保存到: {TABLE_DIR / 'lda_keyword_topic_mean_distribution_k5.csv'}")
    print(f"LDA 岗位主导主题占比已保存到: {TABLE_DIR / 'lda_keyword_dominant_topic_share_k5.csv'}")
    print(f"摘要文件已保存到: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()