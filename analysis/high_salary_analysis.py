# -*- coding: utf-8 -*-
"""
high_salary_analysis.py
用于数据分析岗位招聘文本的高薪岗位特征分析（第一轮：标准解释版，修复 Singular matrix 版本）

当前版本功能：
1. 基于全部有效月薪样本，使用 75 分位数定义高薪岗位
2. 输出高薪阈值、高薪组/非高薪组样本量
3. 构造结构化变量：
   - keyword
   - degree_group
   - experience_group
   - city_tier
   - company_size_tier
4. 构造 9 个核心技能哑变量
5. 输出高薪组与非高薪组的结构化分布表、技能分布表和关键对比图
6. 使用 statsmodels formula API 运行 Logistic 回归
7. 自动剔除导致奇异矩阵的稀疏类别和稀疏技能
8. 输出系数表与 OR（odds ratio）表
9. 输出模型输入预览与基础诊断结果
10. 生成 high_salary_summary.txt

第一轮暂不包含：
- 决策树
- 高薪/非高薪岗位关键词差异
- 高薪/非高薪岗位主题差异
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import PerfectSeparationError


# =========================
# 1. 路径与基础配置
# =========================
BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_PATH = BASE_DIR / "data" / "processed" / "clean_jobs_filtered.csv"

OUTPUT_DIR = BASE_DIR / "output"
TABLE_DIR = OUTPUT_DIR / "high_salary_tables"
FIGURE_DIR = OUTPUT_DIR / "high_salary_figures"
SUMMARY_PATH = TABLE_DIR / "high_salary_summary.txt"

HIGH_SALARY_QUANTILE = 0.75

CORE_KEYWORDS = [
    "数据分析师",
    "BI分析师",
    "用户分析师",
    "经营分析师",
    "商业分析师"
]

CORE_SKILLS = [
    "SQL",
    "Python",
    "Excel",
    "Tableau",
    "Power BI",
    "数据建模",
    "机器学习",
    "数据仓库",
    "ETL",
]

SKILL_COL_MAP = {
    "SQL": "skill_sql",
    "Python": "skill_python",
    "Excel": "skill_excel",
    "Tableau": "skill_tableau",
    "Power BI": "skill_power_bi",
    "数据建模": "skill_data_modeling",
    "机器学习": "skill_machine_learning",
    "数据仓库": "skill_data_warehouse",
    "ETL": "skill_etl",
}

FIRST_TIER_CITIES = {"北京", "上海", "广州", "深圳"}
NEW_FIRST_TIER_CITIES = {
    "成都", "杭州", "重庆", "武汉", "苏州",
    "西安", "南京", "长沙", "郑州", "天津",
    "合肥", "青岛", "东莞", "宁波", "佛山"
}

SMALL_COMPANY = {"20人以下", "20-99人"}
MEDIUM_COMPANY = {"100-299人", "300-499人"}
LARGE_COMPANY = {"500-999人"}
XLARGE_COMPANY = {"1000-9999人", "10000人以上"}

MODEL_REF_MAP = {
    "keyword_model": "数据分析师",
    "degree_group": "本科",
    "experience_group": "1-3年",
    "city_tier": "其他",
    "company_size_model": "微型或小型",
}

MODEL_CAT_COLS = [
    "keyword_model",
    "degree_group",
    "experience_group",
    "city_tier",
    "company_size_model",
]

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


# =========================
# 2. 通用工具函数
# =========================
def ensure_dirs():
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def save_table(df: pd.DataFrame, file_name: str):
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

    required_cols = [
        "salary_unit", "salary_avg", "keyword",
        "degree_std", "experience_std",
        "city", "company_size_raw",
        "skill_keywords_extract"
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"缺少必要字段: {missing_cols}")

    return df


# =========================
# 3. 月薪样本与高薪划分
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


def build_threshold_info(df, threshold):
    total_n = len(df)
    high_n = int(df["is_high_salary"].sum())
    low_n = int((df["is_high_salary"] == 0).sum())

    info_df = pd.DataFrame({
        "metric": [
            "monthly_sample_count",
            "high_salary_threshold_q75",
            "high_salary_count",
            "non_high_salary_count",
            "high_salary_rate",
            "non_high_salary_rate",
        ],
        "value": [
            total_n,
            round(threshold, 2),
            high_n,
            low_n,
            round(high_n / total_n, 4) if total_n else np.nan,
            round(low_n / total_n, 4) if total_n else np.nan,
        ]
    })

    count_df = pd.DataFrame({
        "salary_group": ["高薪岗位", "非高薪岗位"],
        "count": [high_n, low_n],
        "rate": [
            round(high_n / total_n, 4) if total_n else np.nan,
            round(low_n / total_n, 4) if total_n else np.nan,
        ]
    })

    return info_df, count_df


# =========================
# 4. 分层变量构造
# =========================
def build_degree_group(text):
    text = "" if pd.isna(text) else str(text).strip()

    if text == "不限":
        return "不限"
    if text == "大专":
        return "大专"
    if text == "本科":
        return "本科"
    if text in {"硕士", "博士"}:
        return "硕士及以上"
    return "其他/未说明"


def build_experience_group(text):
    text = "" if pd.isna(text) else str(text).strip()

    if text == "不限":
        return "不限"
    if text in {"应届/在校", "1年以内"}:
        return "应届及1年以内"
    if text == "1-3年":
        return "1-3年"
    if text == "3-5年":
        return "3-5年"
    if text in {"5-10年", "10年以上"}:
        return "5年以上"
    return "其他/未说明"


def build_city_tier(city):
    city = "" if pd.isna(city) else str(city).strip()

    if city in FIRST_TIER_CITIES:
        return "一线"
    if city in NEW_FIRST_TIER_CITIES:
        return "新一线"
    return "其他"


def build_company_size_tier(size_raw):
    text = "" if pd.isna(size_raw) else str(size_raw).strip()

    if text in SMALL_COMPANY:
        return "微型或小型"
    if text in MEDIUM_COMPANY:
        return "中型"
    if text in LARGE_COMPANY:
        return "大型"
    if text in XLARGE_COMPANY:
        return "超大型"
    return "其他/未说明"


def add_group_variables(df):
    out = df.copy()
    out["degree_group"] = out["degree_std"].apply(build_degree_group)
    out["experience_group"] = out["experience_std"].apply(build_experience_group)
    out["city_tier"] = out["city"].apply(build_city_tier)
    out["company_size_tier"] = out["company_size_raw"].apply(build_company_size_tier)
    return out


# =========================
# 5. 技能变量构造
# =========================
def extract_skill_set(skill_text):
    if pd.isna(skill_text):
        return set()

    text = str(skill_text).strip()
    if not text:
        return set()

    parts = [x.strip() for x in text.split("|")]
    parts = [x for x in parts if x]
    return set(parts)


def build_skill_dummies(df, skill_list=None):
    if skill_list is None:
        skill_list = CORE_SKILLS

    out = df.copy()
    skill_sets = out["skill_keywords_extract"].apply(extract_skill_set)

    for skill in skill_list:
        col_name = SKILL_COL_MAP[skill]
        out[col_name] = skill_sets.apply(lambda s: int(skill in s))

    return out


# =========================
# 6. 模型专用变量与诊断
# =========================
def build_keyword_model_group(text):
    text = "" if pd.isna(text) else str(text).strip()
    if text in CORE_KEYWORDS:
        return text
    return "其他分析岗位"


def build_company_size_model_group(text):
    text = "" if pd.isna(text) else str(text).strip()
    if text == "其他/未说明":
        return "微型或小型"
    return text


def build_model_dataset(df):
    out = df.copy()

    out["keyword_model"] = out["keyword"].apply(build_keyword_model_group)
    out["company_size_model"] = out["company_size_tier"].apply(build_company_size_model_group)

    keep_cols = [
        "is_high_salary",
        "keyword_model",
        "degree_group",
        "experience_group",
        "city_tier",
        "company_size_model",
    ] + list(SKILL_COL_MAP.values())

    df_model = out[keep_cols].copy()

    # 转成字符串，避免类别残留导致问题
    for col in MODEL_CAT_COLS:
        df_model[col] = df_model[col].astype(str).str.strip()

    return df_model


def save_model_dataset_preview(df_model, n=50):
    preview_df = df_model.head(n).copy()
    save_table(preview_df, "high_salary_model_dataset_preview.csv")


def check_class_balance(df_model):
    balance_df = (
        df_model["is_high_salary"]
        .value_counts(dropna=False)
        .rename_axis("is_high_salary")
        .reset_index(name="count")
    )
    total = balance_df["count"].sum()
    balance_df["rate"] = (balance_df["count"] / total).round(4)
    return balance_df


def check_feature_missing(df_model):
    missing_df = pd.DataFrame({
        "column": df_model.columns,
        "missing_count": df_model.isna().sum().values,
        "missing_rate": (df_model.isna().mean().round(4)).values,
    }).sort_values(["missing_count", "column"], ascending=[False, True]).reset_index(drop=True)
    return missing_df


def diagnose_categorical_balance(df_model):
    records = []
    for col in MODEL_CAT_COLS:
        ct = pd.crosstab(df_model[col], df_model["is_high_salary"], dropna=False)
        for target in [0, 1]:
            if target not in ct.columns:
                ct[target] = 0
        ct = ct[[0, 1]]

        for level, row in ct.iterrows():
            records.append({
                "column": col,
                "level": level,
                "non_high_salary_count": int(row[0]),
                "high_salary_count": int(row[1]),
                "total_count": int(row.sum()),
            })

    return pd.DataFrame(records)


def filter_sparse_levels_for_logit(df_model, min_count=8):
    """
    迭代剔除会导致 Logit 不稳定的类别水平：
    - 总样本数过少
    - 只出现在高薪/非高薪的一边
    参考组不允许被剔除，否则直接报错提醒更换参考组
    """
    out = df_model.copy()
    dropped_records = []

    while True:
        changed = False

        for col in MODEL_CAT_COLS:
            ref = MODEL_REF_MAP[col]
            ct = pd.crosstab(out[col], out["is_high_salary"], dropna=False)

            for target in [0, 1]:
                if target not in ct.columns:
                    ct[target] = 0
            ct = ct[[0, 1]]

            if ref not in ct.index:
                raise ValueError(f"参考组 {col}={ref} 不存在于当前模型样本中，请检查数据或更换参考组。")

            ref_row = ct.loc[ref]
            if int(ref_row.sum()) < min_count or int(ref_row[0]) == 0 or int(ref_row[1]) == 0:
                raise ValueError(
                    f"参考组 {col}={ref} 过于稀疏或只出现在单一组别中，"
                    f"请更换参考组或进一步合并类别。"
                )

            bad_levels = []
            for level, row in ct.iterrows():
                if level == ref:
                    continue

                total = int(row.sum())
                non_high = int(row[0])
                high = int(row[1])

                if total < min_count:
                    bad_levels.append((level, total, non_high, high, "总样本数过少"))
                elif non_high == 0 or high == 0:
                    bad_levels.append((level, total, non_high, high, "只出现在单一组别"))

            if bad_levels:
                changed = True
                bad_level_names = [x[0] for x in bad_levels]

                for level, total, non_high, high, reason in bad_levels:
                    dropped_records.append({
                        "column": col,
                        "dropped_level": level,
                        "total_count": total,
                        "non_high_salary_count": non_high,
                        "high_salary_count": high,
                        "reason": reason,
                    })

                out = out[~out[col].isin(bad_level_names)].copy()

        if not changed:
            break

    if out["is_high_salary"].nunique() < 2:
        raise ValueError("稀疏类别剔除后，高薪组/非高薪组不足两类，无法继续建模。")

    dropped_df = pd.DataFrame(dropped_records)
    return out, dropped_df


def get_valid_skill_cols_for_logit(df_model, skill_cols, min_positive_count=10):
    valid_cols = []
    dropped_records = []

    for col in skill_cols:
        nunique = df_model[col].nunique(dropna=False)
        positive_count = int(df_model[col].sum())

        if nunique < 2:
            dropped_records.append({
                "skill_col": col,
                "positive_count": positive_count,
                "reason": "无变异"
            })
            continue

        if positive_count < min_positive_count:
            dropped_records.append({
                "skill_col": col,
                "positive_count": positive_count,
                "reason": "出现次数过少"
            })
            continue

        ct = pd.crosstab(df_model[col], df_model["is_high_salary"], dropna=False)
        for idx in [0, 1]:
            if idx not in ct.index:
                ct.loc[idx] = 0
        for target in [0, 1]:
            if target not in ct.columns:
                ct[target] = 0
        ct = ct.sort_index().sort_index(axis=1)

        # 只要 2x2 列联表里有 0，就可能导致完全分离/准完全分离
        if (ct.loc[0, 0] == 0 or ct.loc[0, 1] == 0 or ct.loc[1, 0] == 0 or ct.loc[1, 1] == 0):
            dropped_records.append({
                "skill_col": col,
                "positive_count": positive_count,
                "reason": "与高薪分组近乎完全分离"
            })
            continue

        valid_cols.append(col)

    dropped_df = pd.DataFrame(dropped_records)
    return valid_cols, dropped_df


# =========================
# 7. 分布统计
# =========================
def build_categorical_distribution(df, group_col, target_col="is_high_salary"):
    tmp = (
        df.groupby([target_col, group_col], dropna=False)
        .size()
        .reset_index(name="count")
    )

    group_total = (
        tmp.groupby(target_col)["count"]
        .sum()
        .reset_index(name="group_total")
    )

    out = tmp.merge(group_total, on=target_col, how="left")
    out["rate_in_group"] = (out["count"] / out["group_total"]).round(4)
    out["salary_group"] = out[target_col].map({1: "高薪岗位", 0: "非高薪岗位"})

    out = out[[target_col, "salary_group", group_col, "count", "group_total", "rate_in_group"]]
    return out.sort_values([target_col, "count"], ascending=[False, False]).reset_index(drop=True)


def build_skill_distribution(df, skill_cols, target_col="is_high_salary"):
    rows = []

    for target_value, sub_df in df.groupby(target_col):
        group_name = "高薪岗位" if target_value == 1 else "非高薪岗位"
        group_total = len(sub_df)

        for col in skill_cols:
            count = int(sub_df[col].sum())
            rate = round(count / group_total, 4) if group_total else np.nan
            rows.append({
                "is_high_salary": target_value,
                "salary_group": group_name,
                "skill_col": col,
                "count": count,
                "group_total": group_total,
                "rate_in_group": rate,
            })

    out = pd.DataFrame(rows)
    return out.sort_values(["is_high_salary", "rate_in_group"], ascending=[False, False]).reset_index(drop=True)


# =========================
# 8. 绘图函数
# =========================
def save_group_compare_chart(dist_df, category_col, fig_name, title=None):
    pivot_df = dist_df.pivot(index=category_col, columns="salary_group", values="rate_in_group").fillna(0)

    for col in ["非高薪岗位", "高薪岗位"]:
        if col not in pivot_df.columns:
            pivot_df[col] = 0
    pivot_df = pivot_df[["非高薪岗位", "高薪岗位"]]

    ax = pivot_df.plot(kind="bar", figsize=(10, 6))
    if title is not None:
        plt.title(title)
    plt.xlabel(category_col)
    plt.ylabel("组内占比")
    plt.xticks(rotation=30, ha="right")
    plt.legend(title="岗位组别")

    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", fontsize=8)

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / fig_name, dpi=300, bbox_inches="tight")
    plt.close()


def save_skill_compare_chart(skill_dist_df, fig_name, title=None):
    skill_name_map = {v: k for k, v in SKILL_COL_MAP.items()}
    plot_df = skill_dist_df.copy()
    plot_df["skill_name"] = plot_df["skill_col"].map(skill_name_map)

    pivot_df = plot_df.pivot(index="skill_name", columns="salary_group", values="rate_in_group").fillna(0)

    for col in ["非高薪岗位", "高薪岗位"]:
        if col not in pivot_df.columns:
            pivot_df[col] = 0
    pivot_df = pivot_df[["非高薪岗位", "高薪岗位"]]

    ax = pivot_df.plot(kind="barh", figsize=(10, 6))
    if title is not None:
        plt.title(title)
    plt.xlabel("组内占比")
    plt.ylabel("技能")
    plt.legend(title="岗位组别")

    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", fontsize=8)

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / fig_name, dpi=300, bbox_inches="tight")
    plt.close()


# =========================
# 9. Logistic 回归
# =========================
def run_logistic_regression(df_model):
    skill_cols = list(SKILL_COL_MAP.values())
    valid_skill_cols, dropped_skill_df = get_valid_skill_cols_for_logit(df_model, skill_cols, min_positive_count=10)

    formula_terms = [
        "C(keyword_model, Treatment(reference='数据分析师'))",
        "C(degree_group, Treatment(reference='本科'))",
        "C(experience_group, Treatment(reference='1-3年'))",
        "C(city_tier, Treatment(reference='其他'))",
        "C(company_size_model, Treatment(reference='微型或小型'))",
    ] + valid_skill_cols

    formula = "is_high_salary ~ " + " + ".join(formula_terms)

    fit_method = "logit"
    fallback_used = False

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model = smf.logit(formula=formula, data=df_model)
            result = model.fit(disp=False, maxiter=200)
        except (np.linalg.LinAlgError, PerfectSeparationError, ValueError):
            # 兜底：仍然是逻辑回归，只改用 GLM Binomial 拟合
            model = smf.glm(formula=formula, data=df_model, family=sm.families.Binomial())
            result = model.fit(maxiter=200)
            fit_method = "glm_binomial_fallback"
            fallback_used = True

    return result, formula, valid_skill_cols, dropped_skill_df, fit_method, fallback_used


def build_logit_coefficients_table(result):
    params = result.params
    bse = result.bse
    pvals = result.pvalues
    conf = result.conf_int()

    if isinstance(conf, pd.DataFrame):
        conf.columns = ["conf_low", "conf_high"]
    else:
        conf = pd.DataFrame(conf, columns=["conf_low", "conf_high"], index=params.index)

    coef_df = pd.DataFrame({
        "term": params.index,
        "coef": params.values,
        "std_err": bse.values,
        "z_value": (params / bse).values,
        "p_value": pvals.values,
        "conf_low": conf["conf_low"].values,
        "conf_high": conf["conf_high"].values,
    })

    coef_df = coef_df.reset_index(drop=True)
    return coef_df


def build_odds_ratio_table(result):
    coef_df = build_logit_coefficients_table(result).copy()

    coef_df["odds_ratio"] = np.exp(coef_df["coef"])
    coef_df["or_ci_low"] = np.exp(coef_df["conf_low"])
    coef_df["or_ci_high"] = np.exp(coef_df["conf_high"])

    or_df = coef_df[[
        "term", "coef", "p_value",
        "odds_ratio", "or_ci_low", "or_ci_high"
    ]].copy()

    return or_df


# =========================
# 10. 主流程
# =========================
def main():
    ensure_dirs()

    # 1. 读取数据
    df = load_data()
    n_all_filtered = len(df)

    # 2. 保留有效月薪样本
    df_monthly = filter_valid_monthly_salary(df)
    n_monthly = len(df_monthly)

    # 3. 计算高薪阈值并打标签
    threshold = compute_high_salary_threshold(df_monthly, quantile=HIGH_SALARY_QUANTILE)
    df_monthly = label_high_salary_group(df_monthly, threshold)

    threshold_info_df, group_count_df = build_threshold_info(df_monthly, threshold)
    save_table(threshold_info_df, "high_salary_threshold_info.csv")
    save_table(group_count_df, "high_salary_group_counts.csv")

    # 4. 添加分层变量
    df_monthly = add_group_variables(df_monthly)

    # 5. 添加技能变量
    df_monthly = build_skill_dummies(df_monthly)

    # 6. 输出结构化分布表
    keyword_dist_df = build_categorical_distribution(df_monthly, "keyword")
    degree_dist_df = build_categorical_distribution(df_monthly, "degree_group")
    exp_dist_df = build_categorical_distribution(df_monthly, "experience_group")
    city_dist_df = build_categorical_distribution(df_monthly, "city_tier")
    company_dist_df = build_categorical_distribution(df_monthly, "company_size_tier")

    save_table(keyword_dist_df, "high_salary_keyword_distribution.csv")
    save_table(degree_dist_df, "high_salary_degree_distribution.csv")
    save_table(exp_dist_df, "high_salary_experience_distribution.csv")
    save_table(city_dist_df, "high_salary_city_tier_distribution.csv")
    save_table(company_dist_df, "high_salary_company_size_distribution.csv")

    # 7. 输出技能分布表
    skill_cols = list(SKILL_COL_MAP.values())
    skill_dist_df = build_skill_distribution(df_monthly, skill_cols)
    save_table(skill_dist_df, "high_salary_skill_distribution.csv")

    # 8. 作图
    save_group_compare_chart(
        keyword_dist_df,
        category_col="keyword",
        fig_name="high_salary_keyword_distribution.png",
        title="高薪组与非高薪组的岗位类别分布对比"
    )

    save_group_compare_chart(
        city_dist_df,
        category_col="city_tier",
        fig_name="high_salary_city_tier_distribution.png",
        title="高薪组与非高薪组的城市层级分布对比"
    )

    save_group_compare_chart(
        company_dist_df,
        category_col="company_size_tier",
        fig_name="high_salary_company_size_distribution.png",
        title="高薪组与非高薪组的公司规模分布对比"
    )

    save_skill_compare_chart(
        skill_dist_df,
        fig_name="high_salary_skill_distribution.png",
        title="高薪组与非高薪组的核心技能分布对比"
    )

    # 9. 构造模型数据集
    df_model = build_model_dataset(df_monthly)

    # 10. 建模前诊断：列联表
    cat_balance_df = diagnose_categorical_balance(df_model)
    save_table(cat_balance_df, "high_salary_model_categorical_balance.csv")

    # 11. 剔除导致奇异矩阵的稀疏类别水平
    df_model_filtered, dropped_levels_df = filter_sparse_levels_for_logit(df_model, min_count=8)
    if not dropped_levels_df.empty:
        save_table(dropped_levels_df, "high_salary_model_dropped_levels.csv")

    save_model_dataset_preview(df_model_filtered, n=50)

    balance_df = check_class_balance(df_model_filtered)
    missing_df = check_feature_missing(df_model_filtered)

    # 12. Logistic 回归
    result, formula, valid_skill_cols, dropped_skill_df, fit_method, fallback_used = run_logistic_regression(df_model_filtered)

    coef_df = build_logit_coefficients_table(result)
    or_df = build_odds_ratio_table(result)

    save_table(coef_df, "logit_coefficients.csv")
    save_table(or_df, "logit_odds_ratio.csv")

    if not dropped_skill_df.empty:
        save_table(dropped_skill_df, "high_salary_model_dropped_skills.csv")

    # 13. 写 summary
    core_monthly_n = int(df_monthly["keyword"].isin(CORE_KEYWORDS).sum())

    summary_lines = [
        "===== 高薪岗位特征分析模块：第一轮（标准解释版，修复 Singular matrix 版本） =====",
        f"全部过滤后样本量: {n_all_filtered}",
        f"有效月薪样本量: {n_monthly}",
        f"五类核心岗位中的有效月薪样本量: {core_monthly_n}",
        "",
        "高薪岗位定义：",
        f"- 基于全部有效月薪样本的 {int(HIGH_SALARY_QUANTILE * 100)} 分位数划定高薪阈值",
        f"- 高薪阈值（salary_avg）: {threshold:.2f}",
        "",
        "高薪组 / 非高薪组样本量："
    ]

    for _, row in group_count_df.iterrows():
        summary_lines.append(f"- {row['salary_group']}: {int(row['count'])} ({row['rate']:.2%})")

    summary_lines.extend([
        "",
        "结构化变量口径：",
        "- keyword：原岗位类别（分布统计阶段）",
        "- degree_group：不限 / 大专 / 本科 / 硕士及以上 / 其他或未说明",
        "- experience_group：不限 / 应届及1年以内 / 1-3年 / 3-5年 / 5年以上 / 其他或未说明",
        "- city_tier：一线 / 新一线 / 其他",
        "- company_size_tier：微型或小型 / 中型 / 大型 / 超大型 / 其他或未说明",
        "",
        "模型专用变量口径：",
        "- keyword_model：五类核心岗位保留，其余并入“其他分析岗位”",
        "- company_size_model：将“其他/未说明”并入“微型或小型”",
        "",
        "技能变量口径："
    ])

    for skill in CORE_SKILLS:
        summary_lines.append(f"- {skill} -> {SKILL_COL_MAP[skill]}")

    summary_lines.extend([
        "",
        "模型数据集诊断：",
        "1. 高薪组 / 非高薪组平衡情况："
    ])

    for _, row in balance_df.iterrows():
        group_name = "高薪岗位" if int(row["is_high_salary"]) == 1 else "非高薪岗位"
        summary_lines.append(f"- {group_name}: {int(row['count'])} ({row['rate']:.2%})")

    summary_lines.extend([
        "",
        "2. 模型变量缺失情况（前10项）："
    ])

    for _, row in missing_df.head(10).iterrows():
        summary_lines.append(
            f"- {row['column']}: missing_count={int(row['missing_count'])}, missing_rate={row['missing_rate']:.4f}"
        )

    summary_lines.extend([
        "",
        "3. 因稀疏类别而被剔除的水平："
    ])

    if dropped_levels_df.empty:
        summary_lines.append("- 无")
    else:
        for _, row in dropped_levels_df.iterrows():
            summary_lines.append(
                f"- {row['column']} / {row['dropped_level']}: {row['reason']} "
                f"(total={int(row['total_count'])}, non_high={int(row['non_high_salary_count'])}, high={int(row['high_salary_count'])})"
            )

    summary_lines.extend([
        "",
        "4. 因稀疏或分离而被剔除的技能变量："
    ])

    if dropped_skill_df.empty:
        summary_lines.append("- 无")
    else:
        for _, row in dropped_skill_df.iterrows():
            summary_lines.append(
                f"- {row['skill_col']}: {row['reason']} (positive_count={int(row['positive_count'])})"
            )

    summary_lines.extend([
        "",
        "Logistic 回归说明：",
        f"- 拟合方式: {fit_method}",
        f"- 是否使用 GLM Binomial 兜底: {'是' if fallback_used else '否'}",
        "- 实现方式：statsmodels formula API",
        "- 参考组设置：",
        "  * keyword_model：数据分析师",
        "  * degree_group：本科",
        "  * experience_group：1-3年",
        "  * city_tier：其他",
        "  * company_size_model：微型或小型",
        "  * 技能变量：未出现该技能（0）",
        "",
        "最终进入模型的技能变量："
    ])

    if valid_skill_cols:
        for col in valid_skill_cols:
            summary_lines.append(f"- {col}")
    else:
        summary_lines.append("- 无（仅使用结构化变量建模）")

    summary_lines.extend([
        "",
        "Logistic 回归公式：",
        formula,
        "",
        "Logistic 回归显著变量（p < 0.05）预览："
    ])

    sig_df = or_df[or_df["p_value"] < 0.05].copy()
    if sig_df.empty:
        summary_lines.append("- 当前没有 p < 0.05 的显著变量，请进一步检查模型设定与样本分布。")
    else:
        for _, row in sig_df.iterrows():
            summary_lines.append(
                f"- {row['term']}: OR={row['odds_ratio']:.4f}, 95%CI=({row['or_ci_low']:.4f}, {row['or_ci_high']:.4f}), p={row['p_value']:.4f}"
            )

    summary_lines.extend([
        "",
        "当前阶段已完成：",
        "1. 有效月薪样本筛选",
        "2. 高薪阈值（75分位数）计算",
        "3. 高薪组 / 非高薪组样本划分",
        "4. 结构化变量构造（学历、经验、城市层级、公司规模）",
        "5. 核心技能哑变量构造",
        "6. 高薪组与非高薪组的结构化分布对比",
        "7. 高薪组与非高薪组的技能分布对比",
        "8. 模型稀疏类别与稀疏技能诊断",
        "9. Logistic 回归解释模型",
        "",
        "下一步建议：",
        "1. 增加决策树，形成更直观的分裂路径解释",
        "2. 在五类核心岗位范围内，进一步比较高薪组与非高薪组的关键词差异",
        "3. 在五类核心岗位范围内，进一步比较高薪组与非高薪组的主题差异",
    ])

    write_summary(summary_lines)

    # 14. 控制台输出
    print("高薪岗位特征分析第一轮（标准解释版，修复 Singular matrix 版本）运行完成。")
    print(f"全部过滤后样本量: {n_all_filtered}")
    print(f"有效月薪样本量: {n_monthly}")
    print(f"高薪阈值（75分位数）: {threshold:.2f}")
    print(f"最终用于模型的样本量: {len(df_model_filtered)}")
    print(f"拟合方式: {fit_method}")
    print(f"模型输入预览已保存到: {TABLE_DIR / 'high_salary_model_dataset_preview.csv'}")
    print(f"Logit 系数表已保存到: {TABLE_DIR / 'logit_coefficients.csv'}")
    print(f"Logit OR 表已保存到: {TABLE_DIR / 'logit_odds_ratio.csv'}")
    print(f"摘要文件已保存到: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()