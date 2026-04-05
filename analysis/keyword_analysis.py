# -*- coding: utf-8 -*-
"""
keyword_analysis.py
用于数据分析岗位招聘文本的关键词提取：
(1) TF-IDF（sklearn + 自定义分词）
(2) TextRank（jieba.analyse）

当前版本：核心结果版（第一轮）
- 输出全样本 TF-IDF Top30
- 输出分岗位 TF-IDF Top20
- 输出全样本 TextRank Top30
- 输出分岗位 TextRank Top20
- 输出 keyword_summary.txt
"""

from pathlib import Path
import re
import jieba
import jieba.analyse
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


# =========================
# 1. 路径配置
# =========================
BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_PATH = BASE_DIR / "data" / "processed" / "clean_jobs_filtered.csv"
OUTPUT_DIR = BASE_DIR / "output"
TABLE_DIR = OUTPUT_DIR / "keyword_tables"
FIGURE_DIR = OUTPUT_DIR / "keyword_figures"
SUMMARY_PATH = TABLE_DIR / "keyword_summary.txt"

# 如你后面准备单独放停用词文件，可改成这个路径
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

GENERIC_TERMS_FILTER = {
    "数据", "分析", "数据分析", "分析师", "业务",
    "项目", "需求", "问题", "管理", "技术",
    "开发", "设计", "推动"
}

TEXTRANK_ALLOW_POS = ('n', 'nr', 'ns', 'nt', 'nz', 'eng', 'vn')

TEXTRANK_EXTRA_STOPWORDS = {
    "优先", "公司", "问题",
    "能力", "工作", "经验", "团队", "工具", "专业",
    "要求", "岗位", "职责", "任职", "描述",
    "相关", "具有", "具备", "熟悉"
}

TEXTRANK_GENERIC_FILTER = {
    "数据", "分析", "数据分析", "分析师", "业务",
    "项目", "需求", "问题", "管理", "技术",
    "开发", "设计", "推动",
    "流程", "核心", "系统"
}

TEXTRANK_ENTITY_FILTER = {
    "美团"
}

# =========================
# 3. 轻量保护：标准化映射
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
# 4. 自定义招聘场景保护词和停用词
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
# 5. 基础工具函数
# =========================
def ensure_dirs():
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"未找到输入文件: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)

    required_cols = ["keyword", "job_title", "job_desc_raw"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"缺少必要字段: {missing_cols}")

    return df


def filter_core_keywords(df, core_keywords=None):
    if core_keywords is None:
        core_keywords = CORE_KEYWORDS

    df = df.copy()
    df = df[df["keyword"].isin(core_keywords)].copy()
    return df


def build_text_column(df):
    df = df.copy()
    df["job_title"] = df["job_title"].fillna("").astype(str)
    df["job_desc_raw"] = df["job_desc_raw"].fillna("").astype(str)

    df["text"] = (df["job_title"] + " " + df["job_desc_raw"]).str.replace(r"\s+", " ", regex=True).str.strip()
    df["text_norm"] = df["text"].apply(normalize_text)

    return df


def register_protected_words():
    for word in PROTECTED_WORDS:
        jieba.add_word(word, freq=100000)


def load_stopwords(stopwords_path=STOPWORDS_PATH):
    stopwords = set()

    # 先尝试读取外部停用词文件
    if stopwords_path.exists():
        with open(stopwords_path, "r", encoding="utf-8") as f:
            for line in f:
                word = line.strip()
                if word:
                    stopwords.add(word)

    # 再加入项目自定义招聘停用词
    stopwords.update(CUSTOM_STOPWORDS)

    return stopwords


def normalize_text(text):
    if pd.isna(text):
        return ""

    text = str(text)

    # 先做关键词保护替换
    for old, new in sorted(NORMALIZE_MAP.items(), key=lambda x: len(x[0]), reverse=True):
        text = text.replace(old, new)

    # 全角英文字母/符号做轻量归一化
    text = text.replace("ｒ", "r").replace("Ｒ", "r")
    text = text.replace("ａ", "a").replace("Ａ", "a")
    text = text.replace("ｉ", "i").replace("Ｉ", "i")

    # 去掉常见编号噪音
    text = re.sub(r"[①②③④⑤⑥⑦⑧⑨⑩]", " ", text)
    text = re.sub(r"\b[一二三四五六七八九十]+\b", " ", text)
    text = re.sub(r"\b\d+[、.)）]\s*", " ", text)

    # 去掉多余空白
    text = re.sub(r"\s+", " ", text).strip()

    return text


def is_meaningful_token(token, stopwords):
    """
    判断一个词是否保留
    """
    if not token:
        return False

    token = token.strip()
    if not token:
        return False

    if token in stopwords:
        return False

    # 去掉纯数字
    if re.fullmatch(r"\d+", token):
        return False

    # 去掉纯符号或下划线类噪音
    if re.fullmatch(r"[_\W]+", token):
        return False

    # 中文单字一般过滤
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
    df = df.copy()
    df["tokens"] = df["text"].apply(lambda x: tokenize_text(x, stopwords))
    df["text_cut"] = df["tokens"].apply(lambda x: " ".join(x))
    return df


def get_tfidf_global(df, top_n=30):
    """
    全样本 TF-IDF：
    - 输入：预处理后的 df，其中包含 text_cut 列
    - 输出：按平均 TF-IDF 排序后的 TopN 关键词结果
    """
    corpus = df["text_cut"].fillna("").astype(str)
    corpus = corpus[corpus.str.strip() != ""]

    if len(corpus) == 0:
        raise ValueError("text_cut 全为空，无法计算 TF-IDF。")

    vectorizer = TfidfVectorizer(
        tokenizer=str.split,     # 直接使用我们自己分好词的 text_cut
        preprocessor=None,
        token_pattern=None,      # 使用自定义 tokenizer 时必须设为 None
        lowercase=False,         # 前面已经统一过大小写，这里不再重复处理
        ngram_range=(1, 1)       # unigram：只用单词词项
    )

    tfidf_matrix = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names_out()

    # 平均 TF-IDF：对每个词，在所有岗位文本中的 TF-IDF 取平均
    mean_scores = tfidf_matrix.mean(axis=0).A1

    tfidf_df = pd.DataFrame({
        "term": feature_names,
        "mean_tfidf": mean_scores
    })

    tfidf_df = tfidf_df.sort_values(
        by="mean_tfidf",
        ascending=False
    ).reset_index(drop=True)

    tfidf_df["rank"] = range(1, len(tfidf_df) + 1)
    tfidf_df = tfidf_df[["rank", "term", "mean_tfidf"]]

    if top_n is not None:
        return tfidf_df.head(top_n).copy()

    return tfidf_df.copy()


def filter_tfidf_terms(tfidf_df, excluded_terms, top_n=30):
    """
    在 TF-IDF 结果层做二次泛词过滤
    - 不改原始分词结果
    - 只对排序后的词表做筛除
    """
    filtered_df = tfidf_df[~tfidf_df["term"].isin(excluded_terms)].copy()
    filtered_df = filtered_df.reset_index(drop=True)
    filtered_df["rank"] = range(1, len(filtered_df) + 1)

    if top_n is not None:
        return filtered_df.head(top_n).copy()

    return filtered_df.copy()


def get_tfidf_by_keyword(df, group_col="keyword", top_n=20, excluded_terms=None):
    """
    分组 TF-IDF：
    - 每个岗位子集单独 fit 一个 TF-IDF
    - 每组同时保留 raw / filtered 两版结果
    - 返回长表：
      keyword | version | rank | term | mean_tfidf
    """
    result_frames = []

    for group_name, group_df in df.groupby(group_col):
        corpus = group_df["text_cut"].fillna("").astype(str)
        corpus = corpus[corpus.str.strip() != ""]

        if len(corpus) == 0:
            continue

        vectorizer = TfidfVectorizer(
            tokenizer=str.split,
            preprocessor=None,
            token_pattern=None,
            lowercase=False,
            ngram_range=(1, 1)
        )

        tfidf_matrix = vectorizer.fit_transform(corpus)
        feature_names = vectorizer.get_feature_names_out()
        mean_scores = tfidf_matrix.mean(axis=0).A1

        group_tfidf_df = pd.DataFrame({
            "term": feature_names,
            "mean_tfidf": mean_scores
        }).sort_values(
            by="mean_tfidf",
            ascending=False
        ).reset_index(drop=True)

        # raw 版
        raw_df = group_tfidf_df.copy()
        raw_df["rank"] = range(1, len(raw_df) + 1)
        raw_df = raw_df.head(top_n).copy()
        raw_df["keyword"] = group_name
        raw_df["version"] = "raw"
        raw_df = raw_df[["keyword", "version", "rank", "term", "mean_tfidf"]]
        result_frames.append(raw_df)

        # filtered 版
        if excluded_terms is not None:
            filtered_df = group_tfidf_df[~group_tfidf_df["term"].isin(excluded_terms)].copy()
        else:
            filtered_df = group_tfidf_df.copy()

        filtered_df = filtered_df.reset_index(drop=True)
        filtered_df["rank"] = range(1, len(filtered_df) + 1)
        filtered_df = filtered_df.head(top_n).copy()
        filtered_df["keyword"] = group_name
        filtered_df["version"] = "filtered"
        filtered_df = filtered_df[["keyword", "version", "rank", "term", "mean_tfidf"]]
        result_frames.append(filtered_df)

    if not result_frames:
        return pd.DataFrame(columns=["keyword", "version", "rank", "term", "mean_tfidf"])

    return pd.concat(result_frames, ignore_index=True)


def get_textrank_global(
    df,
    text_col="text_norm",
    top_k_per_doc=20,
    top_n=30,
    allow_pos=None,
    stopwords=None,
    extra_stopwords=None,
    entity_filter=None
):
    """
    全样本 TextRank：
    - 对每条岗位文本分别提取关键词
    - 再在全样本层面做汇总
    - 排序规则：先看 doc_freq，再看 avg_weight
    - 在结果层过滤通用停用词 + TextRank 专项停用词
    """
    if allow_pos is None:
        allow_pos = TEXTRANK_ALLOW_POS

    if stopwords is None:
        stopwords = set()
    else:
        stopwords = set(stopwords)

    if extra_stopwords is None:
        extra_stopwords = TEXTRANK_EXTRA_STOPWORDS
    else:
        extra_stopwords = set(extra_stopwords)

    if entity_filter is None:
        entity_filter = TEXTRANK_ENTITY_FILTER
    else:
        entity_filter = set(entity_filter)

    excluded_terms = stopwords | extra_stopwords | entity_filter

    records = []

    texts = df[text_col].fillna("").astype(str)

    for doc_id, text in enumerate(texts):
        text = text.strip()
        if not text:
            continue

        keywords = jieba.analyse.textrank(
            text,
            topK=top_k_per_doc,
            withWeight=True,
            allowPOS=allow_pos
        )

        for term, weight in keywords:
            term = str(term).strip().lower()
            if not term:
                continue
            if term in excluded_terms:
                continue
            if re.fullmatch(r"[_\W]+", term):
                continue
            if len(term) == 1 and re.fullmatch(r"[\u4e00-\u9fff]", term):
                continue

            records.append({
                "doc_id": doc_id,
                "term": term,
                "weight": float(weight)
            })

    if not records:
        return pd.DataFrame(columns=["rank", "term", "doc_freq", "avg_weight", "total_weight"])

    tr_df = pd.DataFrame(records)

    summary_df = (
        tr_df.groupby("term")
        .agg(
            doc_freq=("doc_id", "nunique"),
            avg_weight=("weight", "mean"),
            total_weight=("weight", "sum")
        )
        .reset_index()
        .sort_values(
            by=["doc_freq", "avg_weight"],
            ascending=[False, False]
        )
        .reset_index(drop=True)
    )

    summary_df["rank"] = range(1, len(summary_df) + 1)
    summary_df = summary_df[["rank", "term", "doc_freq", "avg_weight", "total_weight"]]

    if top_n is not None:
        return summary_df.head(top_n).copy()

    return summary_df.copy()


def filter_textrank_terms(textrank_df, excluded_terms, top_n=30):
    """
    在 TextRank 结果层做二次泛词过滤
    - 不改原始 TextRank 提取结果
    - 只对排序后的词表做筛除
    """
    filtered_df = textrank_df[~textrank_df["term"].isin(excluded_terms)].copy()
    filtered_df = filtered_df.reset_index(drop=True)
    filtered_df["rank"] = range(1, len(filtered_df) + 1)

    if top_n is not None:
        return filtered_df.head(top_n).copy()

    return filtered_df.copy()


def get_textrank_by_keyword(
    df,
    group_col="keyword",
    text_col="text_norm",
    top_k_per_doc=20,
    top_n=20,
    allow_pos=None,
    stopwords=None,
    extra_stopwords=None,
    excluded_terms=None,
    entity_filter=None
):
    """
    分组 TextRank：
    - 每个岗位子集分别逐条文本提取 TextRank
    - 再在组内汇总
    - 每组同时保留 raw / filtered 两版结果
    - 返回长表：
      keyword | version | rank | term | doc_freq | avg_weight | total_weight
    """
    if allow_pos is None:
        allow_pos = TEXTRANK_ALLOW_POS

    if stopwords is None:
        stopwords = set()
    else:
        stopwords = set(stopwords)

    if extra_stopwords is None:
        extra_stopwords = TEXTRANK_EXTRA_STOPWORDS
    else:
        extra_stopwords = set(extra_stopwords)

    if entity_filter is None:
        entity_filter = TEXTRANK_ENTITY_FILTER
    else:
        entity_filter = set(entity_filter)

    if excluded_terms is None:
        excluded_terms = TEXTRANK_GENERIC_FILTER
    else:
        excluded_terms = set(excluded_terms)

    result_frames = []

    for group_name, group_df in df.groupby(group_col):
        records = []
        texts = group_df[text_col].fillna("").astype(str).reset_index(drop=True)

        for doc_id, text in enumerate(texts):
            text = text.strip()
            if not text:
                continue

            keywords = jieba.analyse.textrank(
                text,
                topK=top_k_per_doc,
                withWeight=True,
                allowPOS=allow_pos
            )

            for term, weight in keywords:
                term = str(term).strip().lower()
                if not term:
                    continue
                if term in stopwords or term in extra_stopwords or term in entity_filter:
                    continue
                if re.fullmatch(r"[_\W]+", term):
                    continue
                if len(term) == 1 and re.fullmatch(r"[\u4e00-\u9fff]", term):
                    continue

                records.append({
                    "doc_id": doc_id,
                    "term": term,
                    "weight": float(weight)
                })

        if not records:
            continue

        group_tr_df = pd.DataFrame(records)

        summary_df = (
            group_tr_df.groupby("term")
            .agg(
                doc_freq=("doc_id", "nunique"),
                avg_weight=("weight", "mean"),
                total_weight=("weight", "sum")
            )
            .reset_index()
            .sort_values(
                by=["doc_freq", "avg_weight"],
                ascending=[False, False]
            )
            .reset_index(drop=True)
        )

        # raw 版
        raw_df = summary_df.copy()
        raw_df["rank"] = range(1, len(raw_df) + 1)
        raw_df = raw_df[["rank", "term", "doc_freq", "avg_weight", "total_weight"]]
        raw_df = raw_df.head(top_n).copy()
        raw_df["keyword"] = group_name
        raw_df["version"] = "raw"
        raw_df = raw_df[["keyword", "version", "rank", "term", "doc_freq", "avg_weight", "total_weight"]]
        result_frames.append(raw_df)

        # filtered 版
        filtered_df = summary_df[~summary_df["term"].isin(excluded_terms)].copy()
        filtered_df = filtered_df.reset_index(drop=True)
        filtered_df["rank"] = range(1, len(filtered_df) + 1)
        filtered_df = filtered_df[["rank", "term", "doc_freq", "avg_weight", "total_weight"]]
        filtered_df = filtered_df.head(top_n).copy()
        filtered_df["keyword"] = group_name
        filtered_df["version"] = "filtered"
        filtered_df = filtered_df[["keyword", "version", "rank", "term", "doc_freq", "avg_weight", "total_weight"]]
        result_frames.append(filtered_df)

    if not result_frames:
        return pd.DataFrame(columns=[
            "keyword", "version", "rank", "term", "doc_freq", "avg_weight", "total_weight"
        ])

    return pd.concat(result_frames, ignore_index=True)


def save_table(df, file_name):
    save_path = TABLE_DIR / file_name
    df.to_csv(save_path, index=False, encoding="utf-8-sig")


def write_summary(lines):
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(str(line) + "\n")


# =========================
# 6. 主流程（第一阶段先只检查预处理）
# =========================
def main():
    ensure_dirs()
    register_protected_words()

    df = load_data()
    n_raw = len(df)

    df = filter_core_keywords(df, CORE_KEYWORDS)
    n_core = len(df)

    df = build_text_column(df)
    stopwords = load_stopwords()
    df = tokenize_corpus(df, stopwords)

    tfidf_all = get_tfidf_global(df, top_n=None)

    # 原始版 Top30 tfidf：体现总体主题关键词
    tfidf_top30 = tfidf_all.head(30).copy()
    save_table(tfidf_top30, "tfidf_global_top30.csv")

    # 过滤版 Top30 tfidf：体现更有区分度的岗位能力/需求关键词
    tfidf_filtered_top30 = filter_tfidf_terms(
        tfidf_all,
        excluded_terms=GENERIC_TERMS_FILTER,
        top_n=30
    )
    save_table(tfidf_filtered_top30, "tfidf_global_filtered_top30.csv")

    # 分组版 Top30 tfidf
    tfidf_by_keyword = get_tfidf_by_keyword(
        df,
        group_col="keyword",
        top_n=20,
        excluded_terms=GENERIC_TERMS_FILTER
    )
    save_table(tfidf_by_keyword, "tfidf_by_keyword_top20.csv")

    # 全样本 textrank
    textrank_global_all = get_textrank_global(
        df,
        text_col="text_norm",
        top_k_per_doc=20,
        top_n=None,
        allow_pos=TEXTRANK_ALLOW_POS,
        stopwords=stopwords,
        extra_stopwords=TEXTRANK_EXTRA_STOPWORDS,
        entity_filter=TEXTRANK_ENTITY_FILTER
    )

    # 原始版 Top30 textrank：体现总体语义中心
    textrank_global_top30 = textrank_global_all.head(30).copy()
    save_table(textrank_global_top30, "textrank_global_top30.csv")

    # 过滤版 Top30 textrank：体现更有解释价值的岗位主题词
    textrank_global_filtered_top30 = filter_textrank_terms(
        textrank_global_all,
        excluded_terms=TEXTRANK_GENERIC_FILTER,
        top_n=30
    )
    save_table(textrank_global_filtered_top30, "textrank_global_filtered_top30.csv")

    # 分组版 Top20 textrank
    textrank_by_keyword = get_textrank_by_keyword(
        df,
        group_col="keyword",
        text_col="text_norm",
        top_k_per_doc=20,
        top_n=20,
        allow_pos=TEXTRANK_ALLOW_POS,
        stopwords=stopwords,
        extra_stopwords=TEXTRANK_EXTRA_STOPWORDS,
        excluded_terms=TEXTRANK_GENERIC_FILTER,
        entity_filter=TEXTRANK_ENTITY_FILTER
    )
    save_table(textrank_by_keyword, "textrank_by_keyword_top20.csv")

    # 先保存一份预处理预览，便于检查
    preview_cols = ["keyword", "job_title", "text", "tokens", "text_cut"]
    preview_df = df[preview_cols].head(30).copy()
    save_table(preview_df, "keyword_preprocess_preview.csv")

    # 输出各类岗位样本量
    keyword_counts = (
        df["keyword"]
        .value_counts()
        .rename_axis("keyword")
        .reset_index(name="count")
    )
    save_table(keyword_counts, "keyword_group_counts.csv")

    summary_lines = [
        "===== 关键词分析模块：TF-IDF 全样本阶段 =====",
        f"原始样本量: {n_raw}",
        f"进入5类核心岗位后的样本量: {n_core}",
        "",
        "核心岗位样本分布："
    ]

    for _, row in keyword_counts.iterrows():
        summary_lines.append(f"- {row['keyword']}: {row['count']}")

    summary_lines.extend([
        "",
        f"停用词总数: {len(stopwords)}",
        f"预处理预览文件: {TABLE_DIR / 'keyword_preprocess_preview.csv'}",
        f"分组统计文件: {TABLE_DIR / 'keyword_group_counts.csv'}",
        f"TF-IDF 原始版结果文件: {TABLE_DIR / 'tfidf_global_top30.csv'}",
        f"TF-IDF 过滤版结果文件: {TABLE_DIR / 'tfidf_global_filtered_top30.csv'}",
        f"TF-IDF 分组结果文件: {TABLE_DIR / 'tfidf_by_keyword_top20.csv'}",
        f"TextRank 原始版结果文件: {TABLE_DIR / 'textrank_global_top30.csv'}",
        f"TextRank 过滤版结果文件: {TABLE_DIR / 'textrank_global_filtered_top30.csv'}",
        f"TextRank 分组结果文件: {TABLE_DIR / 'textrank_by_keyword_top20.csv'}",
        "",
        "原始版全样本 TF-IDF Top10："
    ])

    for _, row in tfidf_top30.head(10).iterrows():
        summary_lines.append(
            f"{int(row['rank'])}. {row['term']} : {row['mean_tfidf']:.6f}"
        )

    summary_lines.extend([
        "",
        "过滤版全样本 TF-IDF Top10："
    ])

    for _, row in tfidf_filtered_top30.head(10).iterrows():
        summary_lines.append(
            f"{int(row['rank'])}. {row['term']} : {row['mean_tfidf']:.6f}"
        )

    summary_lines.extend([
        "",
        "各岗位 filtered 版 TF-IDF Top5 预览："
    ])

    for kw in CORE_KEYWORDS:
        summary_lines.append("")
        summary_lines.append(f"[{kw}]")
        sub_df = tfidf_by_keyword[
            (tfidf_by_keyword["keyword"] == kw) &
            (tfidf_by_keyword["version"] == "filtered")
        ].head(5)

        for _, row in sub_df.iterrows():
            summary_lines.append(
                f"{int(row['rank'])}. {row['term']} : {row['mean_tfidf']:.6f}"
            )

    summary_lines.extend([
        "",
        "原始版全样本 TextRank Top10："
    ])

    for _, row in textrank_global_top30.head(10).iterrows():
        summary_lines.append(
            f"{int(row['rank'])}. {row['term']} : doc_freq={int(row['doc_freq'])}, avg_weight={row['avg_weight']:.6f}"
        )

    summary_lines.extend([
        "",
        "过滤版全样本 TextRank Top10："
    ])

    for _, row in textrank_global_filtered_top30.head(10).iterrows():
        summary_lines.append(
            f"{int(row['rank'])}. {row['term']} : doc_freq={int(row['doc_freq'])}, avg_weight={row['avg_weight']:.6f}"
        )

    summary_lines.extend([
        "",
        "各岗位 filtered 版 TextRank Top5 预览："
    ])

    for kw in CORE_KEYWORDS:
        summary_lines.append("")
        summary_lines.append(f"[{kw}]")
        sub_df = textrank_by_keyword[
            (textrank_by_keyword["keyword"] == kw) &
            (textrank_by_keyword["version"] == "filtered")
        ].head(5)

        for _, row in sub_df.iterrows():
            summary_lines.append(
                f"{int(row['rank'])}. {row['term']} : doc_freq={int(row['doc_freq'])}, avg_weight={row['avg_weight']:.6f}"
            )

    summary_lines.extend([
        "",
        "当前阶段已完成：",
        "1. 核心岗位筛选",
        "2. 文本字段构造（job_title + job_desc_raw）",
        "3. 通用停用词/自定义停用词加载",
        "4. 文本标准化与分词清洗",
        "5. 全样本 TF-IDF 关键词提取（原始版 + 泛词过滤版）",
        "6. 分岗位 TF-IDF 关键词提取（raw + filtered）",
        "7. 全样本 TextRank 关键词提取（逐条提取后汇总，原始版 + 过滤版）",
        "8. 分岗位 TextRank 关键词提取（raw + filtered）",
    ])

    write_summary(summary_lines)

    print("预处理阶段完成。")
    print(f"原始样本量: {n_raw}")
    print(f"进入5类核心岗位后的样本量: {n_core}")
    print(f"预处理预览已保存到: {TABLE_DIR / 'keyword_preprocess_preview.csv'}")
    print(f"分组统计已保存到: {TABLE_DIR / 'keyword_group_counts.csv'}")
    print(f"摘要文件已保存到: {SUMMARY_PATH}")
    print(f"TF-IDF 全样本结果已保存到: {TABLE_DIR / 'tfidf_global_top30.csv'}")
    print(f"TF-IDF 过滤版结果已保存到: {TABLE_DIR / 'tfidf_global_filtered_top30.csv'}")
    print(f"TF-IDF 分组结果已保存到: {TABLE_DIR / 'tfidf_by_keyword_top20.csv'}")
    print(f"TextRank 原始版结果已保存到: {TABLE_DIR / 'textrank_global_top30.csv'}")
    print(f"TextRank 过滤版结果已保存到: {TABLE_DIR / 'textrank_global_filtered_top30.csv'}")
    print(f"TextRank 分组结果已保存到: {TABLE_DIR / 'textrank_by_keyword_top20.csv'}")


if __name__ == "__main__":
    main()