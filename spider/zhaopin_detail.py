"""
zhaopin_detail.py

单 driver 稳定版

作用：
1. 输入单个智联招聘职位详情页链接
2. 使用外部传入的 driver 请求页面内容
3. 解析职位核心字段
4. 返回结构化字典，供后续批量采集使用
"""

import json
import time
import re
import csv
from pathlib import Path
from datetime import datetime

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service


# 项目根目录：job_nlp_project/
BASE_DIR = Path(__file__).resolve().parent.parent
# driver 路径：job_nlp_project/driver/chromedriver.exe
CHROMEDRIVER_PATH = BASE_DIR / "driver" / "chromedriver.exe"

# 调试 HTML 保存路径
DEBUG_HTML_PATH = BASE_DIR / "data" / "raw" / "sample_detail_selenium.html"
RAW_CSV_PATH = BASE_DIR / "data" / "raw" / "raw_jobs.csv"
SEARCH_HITS_CSV_PATH = BASE_DIR / "data" / "raw" / "search_hits.csv"


def build_driver(headless: bool = False):
    """
    仅供本文件单独测试时使用。
    正式批量采集中，请由 zhaopin_list.py 顶层统一创建并管理 driver。
    """
    options = Options()
    if headless:
        options.add_argument("--headless=new")

    options.add_argument("--window-size=1400,1000")
    options.add_argument("--disable-blink-features=AutomationControlled")

    service = Service(executable_path=str(CHROMEDRIVER_PATH))
    driver = webdriver.Chrome(service=service, options=options)
    driver.set_page_load_timeout(40)
    return driver


def safe_quit_driver(driver):
    """
    安全关闭 driver。
    """
    if driver is None:
        return

    try:
        driver.quit()
    except Exception as e:
        print(f"[WARN] driver.quit() 失败: {e}")

    time.sleep(2)


def save_debug_html(html: str, save_path: Path) -> None:
    """
    将抓取到的 HTML 保存到本地，便于调试
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(html)


def is_security_verification_page(driver) -> bool:
    """
    判断当前页面是否是智联安全验证页。
    """
    title = driver.title or ""
    html = driver.page_source or ""

    signals = [
        "Security Verification",
        "正在验证连接安全性",
        "Protected by Tencent Cloud EdgeOne",
        "请勾选下方复选框",
    ]

    title_hit = any(s.lower() in title.lower() for s in signals)
    html_hit = any(s in html for s in signals)

    return title_hit or html_hit


def safe_get(data: dict, keys: list, default=""):
    """
    按路径安全地从嵌套字典中取值
    例如：safe_get(state, ["jobInfo", "jobDetail", "detailedPosition", "salary"])
    """
    current = data
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current


def first_nonempty(*values):
    """
    返回第一个非空值。
    """
    for v in values:
        if v is None:
            continue
        if isinstance(v, str) and v.strip() == "":
            continue
        if isinstance(v, (list, dict)) and len(v) == 0:
            continue
        return v
    return ""


def normalize_tag_field(value):
    """
    将标签字段统一转成字符串。
    支持：
    - ["数据挖掘", "数据治理"]
    - [{"value": "数据挖掘"}, {"value": "数据治理"}]
    - [{"tag": "本科"}, {"tag": "3-5年"}]
    """
    if not value:
        return ""

    result = []

    if isinstance(value, list):
        for item in value:
            if isinstance(item, str):
                text = item.strip()
                if text:
                    result.append(text)
            elif isinstance(item, dict):
                text = item.get("value") or item.get("tag") or ""
                text = str(text).strip()
                if text:
                    result.append(text)

    elif isinstance(value, str):
        return value.strip()

    dedup = []
    for x in result:
        if x not in dedup:
            dedup.append(x)

    return " | ".join(dedup)


def html_to_text(html_text: str) -> str:
    """
    将带有 <br> 的职位描述转换为纯文本。
    """
    if not html_text:
        return ""
    soup = BeautifulSoup(html_text, "html.parser")
    return soup.get_text("\n", strip=True)


def strip_leading_marker(text: str) -> str:
    """
    去掉一行文本前面的项目符号、编号，如：
    3.计算机类、电子信息类等相关专业
    • 熟练使用 SQL
    """
    if not text:
        return ""

    text = text.strip()

    text = re.sub(r"^[\u2022\u25cf\u25a0\uf0b7•·▪■◆◇★\-\*\s]+", "", text)
    text = re.sub(r"^[\(（]?\d+[\)）\.、]?\s*", "", text)

    return text.strip()


def clean_job_desc_text(text: str) -> str:
    """
    对职位描述做轻量清洗：
    - 清理特殊项目符号
    - 清理多余空白
    - 尽量保留原始语义与换行结构
    """
    if not text:
        return ""

    text = text.replace("\xa0", " ")
    text = text.replace("\uf0b7", "\n")
    text = text.replace("\u2022", "\n")

    text = re.sub(r"(?m)^\s*[oO](?=[\u4e00-\u9fff])", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)

    return text.strip()


def extract_major_requirement(job_desc: str) -> str:
    """
    从职位描述中提取专业要求。
    """
    if not job_desc:
        return ""

    matches = []
    lines = job_desc.splitlines()

    strong_patterns = [
        r"相关专业",
        r"专业优先",
        r"专业背景",
        r"专业要求",
        r"专业不限",
    ]

    major_keywords = [
        "计算机",
        "计算机类",
        "电子信息",
        "电子信息类",
        "数学",
        "数学类",
        "统计学",
        "统计",
        "保险",
        "金融",
        "经济学",
        "信息管理",
        "软件工程",
        "数据科学",
        "人工智能",
    ]

    for line in lines:
        raw_line = line.strip()
        if not raw_line:
            continue

        cleaned_line = strip_leading_marker(raw_line)
        if not cleaned_line:
            continue

        if any(re.search(pattern, cleaned_line) for pattern in strong_patterns):
            cleaned_line = cleaned_line.rstrip("；;。")
            if cleaned_line not in matches:
                matches.append(cleaned_line)
            continue

        has_degree_word = any(x in cleaned_line for x in ["本科", "硕士", "博士", "学历", "统招"])
        has_major_keyword = any(x in cleaned_line for x in major_keywords)

        if has_degree_word and has_major_keyword:
            cleaned_line = cleaned_line.rstrip("；;。")
            if cleaned_line not in matches:
                matches.append(cleaned_line)

    return " | ".join(matches)


def build_empty_result(url: str, keyword: str) -> dict:
    """
    构建一条空的原始岗位记录
    """
    crawl_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return {
        "keyword": keyword,
        "source": "zhaopin",
        "crawl_time": crawl_time,
        "job_title": "",
        "company_name": "",
        "city": "",
        "salary_raw": "",
        "degree_raw": "",
        "experience_raw": "",
        "job_type_raw": "",
        "publish_time_raw": "",
        "company_type_raw": "",
        "company_size_raw": "",
        "welfare_raw": "",
        "skill_tags_raw": "",
        "major_requirement_raw": "",
        "recruit_num_raw": "",
        "job_desc_raw": "",
        "job_link": url,
        "job_id": "",
    }


def append_job_to_csv(record: dict, csv_path: Path) -> None:
    """
    将一条岗位记录追加写入 CSV。
    如果文件为空，则先写表头。
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(record.keys())
    file_exists = csv_path.exists()
    file_is_empty = (not file_exists) or csv_path.stat().st_size == 0

    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if file_is_empty:
            writer.writeheader()

        writer.writerow(record)


def read_existing_job_ids(csv_path: Path) -> set:
    """
    读取 raw_jobs.csv 中已有的 job_id，用于主表去重。
    """
    existing_ids = set()

    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return existing_ids

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            job_id = (row.get("job_id") or "").strip()
            if job_id:
                existing_ids.add(job_id)

    return existing_ids


def append_search_hit(keyword: str, record: dict, csv_path: Path) -> None:
    """
    追加写入搜索命中关系表 search_hits.csv
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    hit_record = {
        "crawl_time": record.get("crawl_time", ""),
        "keyword": keyword,
        "job_id": record.get("job_id", ""),
        "job_title": record.get("job_title", ""),
        "company_name": record.get("company_name", ""),
        "job_link": record.get("job_link", ""),
    }

    fieldnames = list(hit_record.keys())
    file_exists = csv_path.exists()
    file_is_empty = (not file_exists) or csv_path.stat().st_size == 0

    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if file_is_empty:
            writer.writeheader()

        writer.writerow(hit_record)


def save_record_with_d3_strategy(record: dict) -> None:
    """
    D3 策略：
    1. raw_jobs.csv 按 job_id 去重
    2. search_hits.csv 始终记录本次 keyword 命中关系
    """
    job_id = (record.get("job_id") or "").strip()
    keyword = record.get("keyword", "")

    if not job_id:
        print("[WARNING] 当前记录缺少 job_id，跳过主表去重写入，但仍尝试记录搜索命中关系")

    existing_ids = read_existing_job_ids(RAW_CSV_PATH)

    if job_id and job_id not in existing_ids:
        append_job_to_csv(record, RAW_CSV_PATH)
        print(f"[RAW] 已写入主表: {job_id}")
    elif job_id:
        print(f"[RAW] 主表已存在，跳过重复写入: {job_id}")

    append_search_hit(keyword, record, SEARCH_HITS_CSV_PATH)
    print(f"[HIT] 已记录搜索命中: keyword={keyword}, job_id={job_id}")


def extract_initial_state(soup: BeautifulSoup) -> dict:
    """
    从页面中的 <script> 标签里提取 __INITIAL_STATE__ JSON，
    并转成 Python 字典
    """
    scripts = soup.find_all("script")

    for script in scripts:
        script_text = script.string or script.get_text()
        if not script_text:
            continue

        if "__INITIAL_STATE__=" in script_text:
            json_str = script_text.split("__INITIAL_STATE__=", 1)[1].strip()

            if json_str.endswith(";"):
                json_str = json_str[:-1]

            return json.loads(json_str)

    raise ValueError("未找到 __INITIAL_STATE__")


def parse_job_detail(driver, link: str, keyword: str = "", sleep_sec: int = 5) -> dict:
    """
    使用外部传入的共享 driver 解析职位详情页。

    注意：
    - 不在函数内部创建 driver
    - 不在函数内部关闭 driver
    """
    result = build_empty_result(link, keyword)

    driver.get(link)
    time.sleep(sleep_sec)

    print("current_url:", driver.current_url)
    print("title:", driver.title)

    if is_security_verification_page(driver):
        raise ValueError("详情页被安全验证页拦截")

    html = driver.page_source
    save_debug_html(html, DEBUG_HTML_PATH)

    soup = BeautifulSoup(html, "html.parser")
    state = extract_initial_state(soup)

    print("\n成功提取 __INITIAL_STATE__")
    print("state 顶层 keys:", list(state.keys())[:10])

    job_detail = safe_get(state, ["jobInfo", "jobDetail"], default={})
    detailed_position = safe_get(job_detail, ["detailedPosition"], default={})
    detailed_company = safe_get(job_detail, ["detailedCompany"], default={})

    if isinstance(detailed_position, dict):
        print("\ndetailedPosition 部分 keys:")
        print(list(detailed_position.keys())[:20])

    if isinstance(detailed_company, dict):
        print("\ndetailedCompany 部分 keys:")
        print(list(detailed_company.keys())[:20])

    result["job_title"] = first_nonempty(
        safe_get(detailed_position, ["positionName"]),
        safe_get(job_detail, ["name"]),
    )

    result["company_name"] = first_nonempty(
        safe_get(detailed_company, ["companyName"]),
        safe_get(job_detail, ["companyName"]),
    )

    result["city"] = first_nonempty(
        safe_get(job_detail, ["workCity"]),
        safe_get(detailed_position, ["positionWorkCity"]),
    )

    result["salary_raw"] = first_nonempty(
        safe_get(detailed_position, ["salary"]),
        safe_get(job_detail, ["salary60"]),
    )

    result["degree_raw"] = first_nonempty(
        safe_get(detailed_position, ["education"]),
        safe_get(job_detail, ["education"]),
    )

    result["experience_raw"] = first_nonempty(
        safe_get(detailed_position, ["positionWorkingExp"]),
        safe_get(job_detail, ["workingExp"]),
    )

    result["job_type_raw"] = first_nonempty(
        safe_get(detailed_position, ["workType"]),
        safe_get(job_detail, ["emplType"]),
    )

    result["publish_time_raw"] = first_nonempty(
        safe_get(detailed_position, ["positionPublishTime"]),
        safe_get(detailed_position, ["positionUpdateTime"]),
        safe_get(job_detail, ["publishTime"]),
        safe_get(job_detail, ["positionPublishTime"]),
    )

    result["company_type_raw"] = first_nonempty(
        safe_get(detailed_company, ["industryNameLevel"]),
        safe_get(job_detail, ["propertyName"]),
    )

    result["company_size_raw"] = first_nonempty(
        safe_get(detailed_company, ["companySize"]),
        safe_get(job_detail, ["companySize"]),
    )

    result["welfare_raw"] = normalize_tag_field(
        first_nonempty(
            safe_get(detailed_position, ["welfareTags"]),
            safe_get(job_detail, ["welfareTagList"]),
        )
    )

    result["skill_tags_raw"] = normalize_tag_field(
        first_nonempty(
            safe_get(detailed_position, ["labels"]),
            safe_get(job_detail, ["skillLabel"]),
            safe_get(job_detail, ["showSkillTags"]),
        )
    )

    recruit_num = first_nonempty(
        safe_get(detailed_position, ["recruitNumber"]),
        safe_get(job_detail, ["recruitNumber"]),
    )
    result["recruit_num_raw"] = str(recruit_num) if recruit_num != "" else ""

    raw_desc = first_nonempty(
        safe_get(detailed_position, ["description"]),
        safe_get(job_detail, ["jobDesc"]),
    )
    result["job_desc_raw"] = clean_job_desc_text(html_to_text(raw_desc))

    result["job_id"] = first_nonempty(
        safe_get(detailed_position, ["positionNumber"]),
        safe_get(job_detail, ["number"]),
    )

    result["major_requirement_raw"] = extract_major_requirement(result["job_desc_raw"])

    return result


def parse_multiple_jobs(url_keyword_list, headless: bool = False):
    """
    手动测试多条职位链接的解析效果。
    单 driver 复用版。
    """
    results = []
    driver = None

    try:
        driver = build_driver(headless=headless)

        for i, (url, keyword) in enumerate(url_keyword_list, start=1):
            print(f"\n========== 正在解析第 {i} 条 ==========")
            try:
                record = parse_job_detail(driver=driver, link=url, keyword=keyword)
                results.append(record)

                print("解析成功：", record["job_title"], "|", record["company_name"])
                print("job_id:", record["job_id"])
                print("salary_raw:", record["salary_raw"])
                print("publish_time_raw:", record["publish_time_raw"])

            except Exception as e:
                print(f"[ERROR] 第 {i} 条解析失败: {e}")

    finally:
        safe_quit_driver(driver)

    return results


if __name__ == "__main__":
    test_jobs = [
        (
            "https://www.zhaopin.com/jobdetail/CCL1216489020J40800868709.htm?refcode=4019&srccode=401901&preactionid=0b0da1f4-dcd3-49b6-adc1-64897f3894f1",
            "数据分析师"
        ),
        (
            "https://www.zhaopin.com/jobdetail/CC361558410J40699792606.htm?refcode=4019&srccode=401901&preactionid=f9a25943-4fc9-4068-9c87-1910f042cf8f",
            "商业分析师"
        ),
        (
            "https://www.zhaopin.com/jobdetail/CCL1211071560J40926071315.htm?refcode=4019&srccode=401901&preactionid=112ae4de-9099-4c90-82f9-7312faa46515",
            "运营分析师"
        ),
    ]

    records = parse_multiple_jobs(test_jobs, headless=False)

    print("\n========== 测试完成 ==========")
    print(f"成功返回 {len(records)} 条记录")

    for r in records:
        print(r["job_title"], "|", r["company_name"], "|", r["job_id"])