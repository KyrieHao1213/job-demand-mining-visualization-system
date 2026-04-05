from pathlib import Path

# =========================
# 基础路径
# =========================
DASHBOARD_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DASHBOARD_DIR.parent
DATA_DIR = PROJECT_ROOT / 'data'
OUTPUT_DIR = PROJECT_ROOT / 'output'

BASE_JOBS_PATH = DATA_DIR / 'processed' / 'clean_jobs_filtered.csv'

EDA_TABLE_DIR = OUTPUT_DIR / 'eda_tables'
KEYWORD_TABLE_DIR = OUTPUT_DIR / 'keyword_tables'
LDA_TABLE_DIR = OUTPUT_DIR / 'lda_tables'
HIGH_SALARY_TABLE_DIR = OUTPUT_DIR / 'high_salary_tables'
HIGH_SALARY_KEYWORD_TABLE_DIR = OUTPUT_DIR / 'high_salary_keyword_tables'

# =========================
# 页面基础配置
# =========================
APP_TITLE = '基于招聘信息文本的数据分析岗位需求挖掘与可视化系统'
APP_SUBTITLE = '岗位画像｜技能需求｜文本挖掘｜高薪特征'
PAGE_ICON = '📊'
LAYOUT = 'wide'

CORE_KEYWORDS = [
    '数据分析师',
    'BI分析师',
    '用户分析师',
    '经营分析师',
    '商业分析师',
]

HIGH_SALARY_KEYWORD_TABS = ['数据分析师', '商业分析师']

# 高薪页说明：阈值固定展示，不随筛选动态变化
FIXED_HIGH_SALARY_THRESHOLD = True

# =========================
# 主题颜色
# =========================
PRIMARY_COLOR = '#2F80ED'
SECONDARY_COLOR = '#56CCF2'
ACCENT_COLOR = '#F2994A'
SUCCESS_COLOR = '#27AE60'
TEXT_MUTED = '#94A3B8'
CARD_BG = '#111827'
PAGE_BG = '#0B1220'

SALARY_GROUP_COLOR_MAP = {
    '高薪岗位': '#F2994A',
    '非高薪岗位': '#2F80ED',
}

TOPIC_COLOR_MAP = {
    1: '#2F80ED',
    2: '#56CCF2',
    3: '#27AE60',
    4: '#9B51E0',
    5: '#F2994A',
}

# =========================
# 数据注册表
# =========================
DATASET_REGISTRY = {
    'base_jobs': BASE_JOBS_PATH,
    'keyword_dist': EDA_TABLE_DIR / 'keyword_distribution.csv',
    'city_top15': EDA_TABLE_DIR / 'city_top15_distribution.csv',
    'degree_dist': EDA_TABLE_DIR / 'degree_distribution.csv',
    'exp_dist': EDA_TABLE_DIR / 'experience_distribution.csv',
    'salary_mean': EDA_TABLE_DIR / 'mean_salary_by_keyword.csv',
    'skill_top20': EDA_TABLE_DIR / 'skill_top20_distribution.csv',
    'skill_heatmap': EDA_TABLE_DIR / 'keyword_skill_heatmap.csv',
    'salary_summary_by_exp': EDA_TABLE_DIR / 'salary_summary_by_experience.csv',
    'tfidf_global': KEYWORD_TABLE_DIR / 'tfidf_global_filtered_top30.csv',
    'textrank_global': KEYWORD_TABLE_DIR / 'textrank_global_filtered_top30.csv',
    'tfidf_by_keyword': KEYWORD_TABLE_DIR / 'tfidf_by_keyword_top20.csv',
    'textrank_by_keyword': KEYWORD_TABLE_DIR / 'textrank_by_keyword_top20.csv',
    'lda_perplexity': LDA_TABLE_DIR / 'lda_perplexity_by_k.csv',
    'lda_topics': LDA_TABLE_DIR / 'lda_topics_by_k.csv',
    'high_salary_info': HIGH_SALARY_TABLE_DIR / 'high_salary_threshold_info.csv',
    'high_salary_counts': HIGH_SALARY_TABLE_DIR / 'high_salary_group_counts.csv',
    'high_salary_city': HIGH_SALARY_TABLE_DIR / 'high_salary_city_tier_distribution.csv',
    'high_salary_company': HIGH_SALARY_TABLE_DIR / 'high_salary_company_size_distribution.csv',
    'high_salary_keyword': HIGH_SALARY_TABLE_DIR / 'high_salary_keyword_distribution.csv',
    'high_salary_skill': HIGH_SALARY_TABLE_DIR / 'high_salary_skill_distribution.csv',
    'logit_or': HIGH_SALARY_TABLE_DIR / 'logit_odds_ratio.csv',
    'overall_diff': HIGH_SALARY_KEYWORD_TABLE_DIR / 'overall_tfidf_diff_top15.csv',
    'data_diff': HIGH_SALARY_KEYWORD_TABLE_DIR / 'data_analyst_tfidf_diff_top15.csv',
    'biz_diff': HIGH_SALARY_KEYWORD_TABLE_DIR / 'business_analyst_tfidf_diff_top15.csv',
}


# =========================
# 数据口径说明（与论文/报告口径保持一致）
# =========================
SAMPLE_SCOPE_INFO = {
    'raw_total': 2536,
    'filtered_total': 1125,
    'monthly_total': 1026,
    'core_keyword_desc': '五类核心岗位（数据分析师 / BI分析师 / 用户分析师 / 经营分析师 / 商业分析师）',
    'high_salary_threshold': '14000 元/月',
}
