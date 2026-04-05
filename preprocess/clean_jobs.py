import re
import pandas as pd
import unicodedata
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_JOBS_PATH = BASE_DIR / "data" / "raw" / "raw_jobs.csv"
CLEAN_JOBS_PATH = BASE_DIR / "data" / "processed" / "clean_jobs.csv"
CLEAN_FILTERED_PATH = BASE_DIR / "data" / "processed" / "clean_jobs_filtered.csv"


def normalize_degree(text: str) -> str:
    if pd.isna(text):
        return "其他/未说明"

    text = str(text).strip()

    if "不限" in text:
        return "不限"
    if "博士" in text:
        return "博士"
    if "硕士" in text or "研究生" in text:
        return "硕士"
    if "本科" in text:
        return "本科"
    if "大专" in text or "专科" in text:
        return "大专"

    return "其他/未说明"


def normalize_experience(text: str) -> str:
    if pd.isna(text):
        return "其他/未说明"

    text = str(text).strip()

    if "不限" in text:
        return "不限"
    if "应届" in text or "在校" in text or "实习" in text:
        return "应届/在校"
    if "1年以内" in text or "一年以内" in text:
        return "1年以内"
    if "1-3年" in text:
        return "1-3年"
    if "3-5年" in text:
        return "3-5年"
    if "5-10年" in text:
        return "5-10年"
    if "10年以上" in text:
        return "10年以上"

    return "其他/未说明"


def parse_salary(text: str):
    """
    返回：
    salary_min, salary_max, salary_avg, salary_unit, salary_months
    """
    if pd.isna(text):
        return None, None, None, "", None

    text = str(text).strip()
    if not text:
        return None, None, None, "", None

    salary_months = 12
    month_match = re.search(r"(\d{1,2})薪", text)
    if month_match:
        salary_months = int(month_match.group(1))

    main_text = re.split(r"·\d{1,2}薪", text)[0].strip()
    main_text = re.sub(r"\s+", "", main_text)

    m_day_yuan = re.fullmatch(r"(\d+(?:\.\d+)?)\-(\d+(?:\.\d+)?)元/天", main_text)
    if m_day_yuan:
        low = float(m_day_yuan.group(1))
        high = float(m_day_yuan.group(2))
        avg = (low + high) / 2
        return low, high, avg, "日薪", salary_months

    m_day_plain = re.fullmatch(r"(\d+(?:\.\d+)?)\-(\d+(?:\.\d+)?)/天", main_text)
    if m_day_plain:
        low = float(m_day_plain.group(1))
        high = float(m_day_plain.group(2))
        avg = (low + high) / 2
        return low, high, avg, "日薪", salary_months

    m_year_wan = re.fullmatch(r"(\d+(?:\.\d+)?)\-(\d+(?:\.\d+)?)万/年", main_text)
    if m_year_wan:
        low = float(m_year_wan.group(1)) * 10000
        high = float(m_year_wan.group(2)) * 10000
        avg = (low + high) / 2
        return low, high, avg, "年薪", salary_months

    m_year_yuan = re.fullmatch(r"(\d+(?:\.\d+)?)\-(\d+(?:\.\d+)?)元/年", main_text)
    if m_year_yuan:
        low = float(m_year_yuan.group(1))
        high = float(m_year_yuan.group(2))
        avg = (low + high) / 2
        return low, high, avg, "年薪", salary_months

    m_year_wan_alt = re.fullmatch(r"(\d+(?:\.\d+)?)\-(\d+(?:\.\d+)?)万年薪", main_text)
    if m_year_wan_alt:
        low = float(m_year_wan_alt.group(1)) * 10000
        high = float(m_year_wan_alt.group(2)) * 10000
        avg = (low + high) / 2
        return low, high, avg, "年薪", salary_months

    m_month_wan = re.fullmatch(r"(\d+(?:\.\d+)?)\-(\d+(?:\.\d+)?)万", main_text)
    if m_month_wan:
        low = float(m_month_wan.group(1)) * 10000
        high = float(m_month_wan.group(2)) * 10000
        avg = (low + high) / 2
        return low, high, avg, "月薪", salary_months

    m_month_yuan = re.fullmatch(r"(\d+(?:\.\d+)?)\-(\d+(?:\.\d+)?)元", main_text)
    if m_month_yuan:
        low = float(m_month_yuan.group(1))
        high = float(m_month_yuan.group(2))
        avg = (low + high) / 2
        return low, high, avg, "月薪", salary_months

    m_single_wan = re.fullmatch(r"(\d+(?:\.\d+)?)万", main_text)
    if m_single_wan:
        value = float(m_single_wan.group(1)) * 10000
        return value, value, value, "月薪", salary_months

    m_single_yuan = re.fullmatch(r"(\d+(?:\.\d+)?)元", main_text)
    if m_single_yuan:
        value = float(m_single_yuan.group(1))
        return value, value, value, "月薪", salary_months

    return None, None, None, "", salary_months


def normalize_job_title(text: str) -> str:
    """
    对岗位标题做轻量标准化：
    - 去首尾空格
    - 压缩多余空格
    - 统一 BI 大小写
    """
    if pd.isna(text):
        return ""

    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\bbi\b", "BI", text, flags=re.IGNORECASE)

    return text


def normalize_title(text: str) -> str:
    text = "" if text is None else str(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.strip().lower()
    text = re.sub(r"\s+", "", text)
    return text


def split_title_parts(title: str):
    """
    返回：
    full  : 完整归一化标题
    main  : 去掉所有括号内容后的主标题
    extra : 所有括号内容拼接后的补充信息
    """
    full = normalize_title(title)
    # 提取所有括号中的内容
    extra_list = re.findall(r"[（(](.*?)[）)]", full)
    extra = "".join(extra_list)
    # main 不再取“第一个括号前”
    # 而是删除所有括号内容后得到主标题
    main = re.sub(r"[（(].*?[）)]", "", full)
    return full, main, extra


def contains_any(text: str, keywords) -> bool:
    return any(k in text for k in keywords)


def regex_any(text: str, patterns) -> bool:
    return any(re.search(p, text) for p in patterns)


def title_filter_t1(row):
    """
    title_filter_t1_v1_1
    说明：
    - 先硬负向过滤
    - 再按岗位语义簇保留
    - 主标题优先，括号信息辅助
    - 当前采用偏严格口径，不保留“工程师/分析师”混合岗
    """

    """
    返回：
    title_keep, title_filter_reason

    设计原则：
    1. 先硬负向过滤：明显不是目标分析岗的标题直接剔除
    2. 再按“岗位语义簇”保留：不依赖单个 keyword 分支
    3. 优先看主标题 main，括号内容 extra 只做辅助
    """
    keyword = str(row.get("keyword", "")).strip()
    job_title = str(row.get("job_title", "")).strip()

    full, main, extra = split_title_parts(job_title)
    title_lower_raw = job_title.lower()

    # =========================================================
    # 0) 硬负向过滤：这些即使带“分析”也大概率不是你的目标岗位
    # =========================================================
    hard_drop_rules = [
        (r"舆情", "drop_舆情类"),
        (r"专利", "drop_专利类"),
        (r"招商主管|招商经理|招商总监|招商专员", "drop_招商岗"),
        (r"物业总监|物业经理|物业岗", "drop_物业岗"),
        (r"氛围岗", "drop_氛围岗"),
        (r"数据标注|标注员", "drop_数据标注"),
        (r"日语.*实习", "drop_日语实习"),
        (r"测绘.*实习", "drop_测绘实习"),
        (r"人力资源.*实习", "drop_人资实习"),
        (r"交易员.*实习", "drop_交易员实习"),
        (r"交付.*实习", "drop_交付实习"),
        (r"5g.*实习", "drop_5g实习"),
        (r"网络交付", "drop_交付类"),
        (r"项目总经理|商业项目总经理", "drop_管理岗"),
    ]

    for pat, reason in hard_drop_rules:
        if re.search(pat, full):
            return 0, reason

    # BI/商业智能里的明显非分析岗，单独提前挡掉
    if (("bi" in title_lower_raw) or ("商业智能" in full)) and regex_any(
        full + " " + title_lower_raw,
        [r"开发", r"工程师", r"实施", r"顾问", r"产品", r"运维"]
    ):
        return 0, "drop_bi非分析岗"

    # =========================================================
    # 1) 分析岗的主干判定：没有“分析/挖掘/洞察/研究”主干，一般不过
    # =========================================================
    analysis_core = (
        ("分析" in main)
        or ("数据挖掘" in main)
        or ("商业洞察" in main)
        or ("用户研究" in main)
    )

    if not analysis_core:
        # 对“数据运营与分析实习生”这类并列标题做补充
        if not ("实习" in main and ("分析" in full or "数据挖掘" in full)):
            return 0, "drop_无分析主干"

    # =========================================================
    # 2) 分语义簇保留（核心部分）
    # =========================================================

    # 2.1 数据分析簇
    # 例如：数据分析师 / 高级数据分析师 / 跨境电商数据分析师 /
    #      经营数据分析师 / 数据运营与分析实习生 / 行业分析与数据挖掘实习生
    if (
        ("数据" in main and "分析" in main)
        or ("数据挖掘" in main)
        or ("挖掘" in main and "实习" in main)
        or ("数据运营" in main and "分析" in main)
    ):
        return 1, "keep_数据分析簇"

    # 2.2 用户分析簇
    # 例如：用户分析师 / 用户增长数据分析师 / 用户研究分析师 /
    #      用户体验分析师（偏增长 / 运营方向）
    if (
        ("用户" in main and ("分析" in main or "研究" in main))
        or ("增长" in main and "分析" in main)
        or ("留存" in main and "分析" in main)
        or ("转化" in main and "分析" in main)
        or ("客群" in main and "分析" in main)
        or ("人群" in main and "分析" in main)
    ):
        return 1, "keep_用户分析簇"

    # 2.3 商业 / 商务 / 业务 / 经营分析簇
    # 例如：商业分析师 / 商业洞察分析师 / 商务分析师 / 业务分析师 /
    #      经营分析师 / 团购经营分析师 / 经营数据分析师
    if (
        ("商业" in main and ("分析" in main or "洞察" in main))
        or ("商务" in main and "分析" in main)
        or ("业务" in main and "分析" in main)
        or ("经营" in main and "分析" in main)
        or ("经营数据" in main and "分析" in main)
    ):
        return 1, "keep_商业经营分析簇"

    # 2.4 运营分析簇
    # 例如：运营分析师 / 运营财务分析师 / 用户体验分析师（偏增长 / 运营方向）
    # 注意：这里主标题优先，避免把“战略分析师（运营优化）”误收
    if (
        ("运营" in main and "分析" in main)
        or ("运营财务分析师" in main)
    ):
        return 1, "keep_运营分析簇"

    # 2.5 BI / 商业智能分析簇
    # 例如：BI分析师 / 高级BI分析师 / 商业智能分析师 / BI数据分析师
    if (
        (("bi" in title_lower_raw) and ("分析" in main))
        or ("商业智能" in main and "分析" in main)
    ):
        return 1, "keep_bi分析簇"

    # 2.6 分析类实习簇
    # 例如：数据分析实习生 / 数据运营与分析实习生 / 行业分析与数据挖掘实习生
    if "实习" in main:
        if (
            ("分析" in main and contains_any(main, ["数据", "用户", "商业", "商务", "业务", "经营", "运营"]))
            or ("数据挖掘" in main)
            or (("bi" in title_lower_raw or "商业智能" in main) and "分析" in main)
        ):
            return 1, "keep_分析类实习簇"

    # =========================================================
    # 3) keyword 辅助兜底
    # 作用：避免某些标题表达不标准，但和原搜索词高度一致时被漏掉
    # =========================================================
    if keyword == "商业分析师" and ("商业" in full and ("分析" in full or "洞察" in full)):
        return 1, "keep_keyword_商业分析师"

    if keyword == "用户分析师" and ("用户" in full and ("分析" in full or "研究" in full)):
        return 1, "keep_keyword_用户分析师"

    if keyword == "商务分析师" and (
        ("商务" in full and "分析" in full)
        or ("业务" in full and "分析" in full)
    ):
        return 1, "keep_keyword_商务分析师"

    if keyword == "经营分析师" and ("经营" in full and "分析" in full):
        return 1, "keep_keyword_经营分析师"

    if keyword == "运营分析师" and ("运营" in full and "分析" in full):
        return 1, "keep_keyword_运营分析师"

    if keyword == "数据分析师" and (
        ("数据" in full and "分析" in full) or ("数据挖掘" in full)
    ):
        return 1, "keep_keyword_数据分析师"

    if keyword == "数据分析实习生" and (
        "实习" in full and (("数据" in full and "分析" in full) or ("数据挖掘" in full))
    ):
        return 1, "keep_keyword_数据分析实习生"

    if keyword in ["BI分析师", "商业智能分析师"] and (
        (("bi" in title_lower_raw) and "分析" in full)
        or ("商业智能" in full and "分析" in full)
    ):
        return 1, "keep_keyword_bi分析师"

    # =========================================================
    # 4) 默认过滤
    # =========================================================
    return 0, "drop_no_match"


SKILL_PATTERNS = {
    "Python": [r"(?<![A-Za-z])Python(?![A-Za-z])"],
    "SQL": [r"(?<![A-Za-z])SQL(?![A-Za-z])"],
    "Excel": [r"(?<![A-Za-z])Excel(?![A-Za-z])"],

    "Java": [r"(?<![A-Za-z])Java(?![A-Za-z])"],
    "C++": [r"C\+\+", r"C＋＋"],
    "C#": [r"C#", r"C＃"],
    "JavaScript": [r"(?<![A-Za-z])JavaScript(?![A-Za-z])", r"(?<![A-Za-z])JS(?![A-Za-z])"],
    "Go": [r"(?<![A-Za-z])Go(?![A-Za-z])", r"(?<![A-Za-z])Golang(?![A-Za-z])"],
    "MATLAB": [r"(?<![A-Za-z])MATLAB(?![A-Za-z])"],
    "Shell": [r"(?<![A-Za-z])Shell(?![A-Za-z])"],
    "Bash": [r"(?<![A-Za-z])Bash(?![A-Za-z])"],

    "Tableau": [r"(?<![A-Za-z])Tableau(?![A-Za-z])"],
    "Power BI": [r"(?<![A-Za-z])Power\s*BI(?![A-Za-z])", r"(?<![A-Za-z])PowerBI(?![A-Za-z])"],
    "FineBI": [r"(?<![A-Za-z])FineBI(?![A-Za-z])"],

    "SAS": [r"(?<![A-Za-z])SAS(?![A-Za-z])"],
    "R": [r"(?<![A-Za-z])R(?![A-Za-z])"],

    "Hadoop": [r"(?<![A-Za-z])Hadoop(?![A-Za-z])"],
    "Spark": [r"(?<![A-Za-z])Spark(?![A-Za-z])"],
    "PySpark": [r"(?<![A-Za-z])PySpark(?![A-Za-z])"],
    "Hive": [r"(?<![A-Za-z])Hive(?![A-Za-z])"],
    "ClickHouse": [r"(?<![A-Za-z])ClickHouse(?![A-Za-z])"],
    "MySQL": [r"(?<![A-Za-z])MySQL(?![A-Za-z])"],
    "PostgreSQL": [r"(?<![A-Za-z])PostgreSQL(?![A-Za-z])", r"(?<![A-Za-z])Postgres(?![A-Za-z])"],
    "Oracle": [r"(?<![A-Za-z])Oracle(?![A-Za-z])"],

    "ETL": [r"(?<![A-Za-z])ETL(?![A-Za-z])"],
    "数据仓库": [r"数据仓库"],
    "A/B测试": [r"A/B", r"A-B", r"AB实验", r"A/B实验", r"A/B测试"],
    "机器学习": [r"机器学习"],
    "深度学习": [r"深度学习"],
    "数据治理": [r"数据治理"],
    "数据建模": [r"数据建模", r"建模方法论"],
    "数据可视化": [r"数据可视化"],
    "报表开发": [r"报表开发", r"仪表盘"],
}


def extract_skill_keywords(text: str) -> str:
    """
    从职位描述中提取技能关键词。
    第一版采用规则匹配，返回去重后的技能字符串。
    """
    if pd.isna(text):
        return ""

    text = str(text).strip()
    if not text:
        return ""

    matched_skills = []

    for skill_name, patterns in SKILL_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text, flags=re.IGNORECASE):
                if skill_name not in matched_skills:
                    matched_skills.append(skill_name)
                break

    return " | ".join(matched_skills)


def main():
    if not RAW_JOBS_PATH.exists():
        raise FileNotFoundError(f"找不到原始数据文件：{RAW_JOBS_PATH}")

    df = pd.read_csv(RAW_JOBS_PATH)
    print("原始数据条数：", len(df))

    if df.empty:
        print("raw_jobs.csv 为空，程序结束。")
        return

    required_cols = [
        "job_id", "keyword", "job_title", "city",
        "degree_raw", "experience_raw", "salary_raw", "job_desc_raw"
    ]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise KeyError(f"raw_jobs.csv 缺少必要字段：{missing_cols}")

    # 1. 按 job_id 去重
    df = df.drop_duplicates(subset=["job_id"]).copy()
    print("按 job_id 去重后条数：", len(df))

    # 2. 是否武汉
    df["is_wuhan"] = df["city"].fillna("").astype(str).str.contains("武汉").astype(int)

    # 3. 学历标准化
    df["degree_std"] = df["degree_raw"].apply(normalize_degree)

    # 4. 经验标准化
    df["experience_std"] = df["experience_raw"].apply(normalize_experience)

    # 5. 薪资标准化
    salary_parsed = df["salary_raw"].apply(parse_salary)
    salary_df = pd.DataFrame(
        salary_parsed.tolist(),
        columns=["salary_min", "salary_max", "salary_avg", "salary_unit", "salary_months"],
        index=df.index
    )
    df = pd.concat([df, salary_df], axis=1)

    # 6. 岗位标题标准化与宽松过滤
    df["job_title_norm"] = df["job_title"].apply(normalize_job_title)
    df[["title_keep", "title_filter_reason"]] = df.apply(
        title_filter_t1,
        axis=1,
        result_type="expand"
    )

    # 7. 从职位描述中提取技能关键词
    df["skill_keywords_extract"] = df["job_desc_raw"].apply(extract_skill_keywords)

    # 8. 过滤后的分析表
    df_filtered = df[df["title_keep"] == 1].copy()

    # 输出
    CLEAN_JOBS_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_JOBS_PATH, index=False, encoding="utf-8-sig")
    df_filtered.to_csv(CLEAN_FILTERED_PATH, index=False, encoding="utf-8-sig")

    print("清洗后文件已保存：", CLEAN_JOBS_PATH)
    print("过滤后分析表已保存：", CLEAN_FILTERED_PATH)

    print("\n标题过滤统计：")
    print(df["title_keep"].value_counts(dropna=False))

    print("\n标题过滤原因统计（前20项）：")
    print(df["title_filter_reason"].value_counts(dropna=False).head(20))

    print("\n被过滤掉的岗位标题示例：")
    print(df[df["title_keep"] == 0][["keyword", "job_title", "company_name"]].head(20))

    print("\n技能提取预览：")
    preview_cols = ["job_title", "skill_keywords_extract"]
    if "skill_tags_raw" in df_filtered.columns:
        preview_cols.insert(1, "skill_tags_raw")
    print(df_filtered[preview_cols].head(10))

    print(df["title_keep"].value_counts(dropna=False))
    print(df["title_filter_reason"].value_counts(dropna=False).head(30))
    print(df[df["title_keep"] == 0][["keyword", "job_title", "company_name"]].head(30))
    print(df[df["title_keep"] == 1][["keyword", "job_title", "company_name"]].head(30))


if __name__ == "__main__":
    main()