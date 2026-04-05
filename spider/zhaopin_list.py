"""
zhaopin_list.py

单 driver 稳定版

作用：
1. 输入关键词，打开智联招聘搜索结果页
2. 提取当前搜索结果页中的职位详情链接
3. 复用同一个 driver 解析详情页
4. 使用 D3 策略落盘
"""

import re
import time
from pathlib import Path
from urllib.parse import urljoin

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service

from zhaopin_detail import parse_job_detail, save_record_with_d3_strategy


BASE_DIR = Path(__file__).resolve().parent.parent
CHROMEDRIVER_PATH = BASE_DIR / "driver" / "chromedriver.exe"

RUNTIME_DIR = BASE_DIR / "runtime"
CHROME_PROFILE_DIR = RUNTIME_DIR / "chrome_profile"

DEBUG_LIST_HTML_PATH = BASE_DIR / "data" / "raw" / "sample_list_selenium.html"


def build_driver(headless: bool = False):
    """
    整个批次只创建一个 driver，并复用固定 profile。
    """
    CHROME_PROFILE_DIR.mkdir(parents=True, exist_ok=True)

    options = Options()
    if headless:
        options.add_argument("--headless=new")

    options.add_argument("--window-size=1400,1000")
    options.add_argument("--disable-blink-features=AutomationControlled")

    # 关键：固定绝对路径 profile
    options.add_argument(f"--user-data-dir={str(CHROME_PROFILE_DIR.resolve())}")
    options.add_argument("--profile-directory=Default")

    # 尽量降低自动化痕迹
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)

    service = Service(executable_path=str(CHROMEDRIVER_PATH))
    driver = webdriver.Chrome(service=service, options=options)

    driver.set_page_load_timeout(40)
    return driver


def safe_quit_driver(driver):
    """
    安全关闭 driver，避免异常中断时 profile 文件仍被占用。
    """
    if driver is None:
        return

    try:
        driver.quit()
    except Exception as e:
        print(f"[WARN] driver.quit() 失败: {e}")

    # 给 Chrome 一点时间释放 profile 文件
    time.sleep(2)


def save_debug_html(html: str, save_path: Path) -> None:
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


def extract_job_links(soup: BeautifulSoup) -> list:
    """
    从列表页中提取职位详情链接。
    规则：
    - 扫描所有 <a href=...>
    - 过滤包含 /jobdetail/ 且含 .htm 的链接
    - 去重
    """
    links = []
    seen = set()

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()

        if "/jobdetail/" in href and ".htm" in href:
            full_url = urljoin("https://www.zhaopin.com", href)
            full_url = full_url.replace("http://", "https://")

            if full_url not in seen:
                seen.add(full_url)
                links.append(full_url)

    return links


def build_paged_url(base_url: str, page_num: int) -> str:
    """
    将列表页 URL 中的 /p1 替换为指定页码 /p{page_num}
    兼容两种情况：
    1. .../p1
    2. .../p1?srccode=401801
    """
    if page_num < 1:
        raise ValueError("page_num 必须 >= 1")

    if re.search(r"/p\d+(\?|$)", base_url):
        return re.sub(
            r"/p\d+(\?|$)",
            lambda m: f"/p{page_num}{m.group(1)}",
            base_url
        )

    raise ValueError(f"URL 中未找到页码模式 /p数字 : {base_url}")


def open_list_page_and_get_links(driver, url: str, sleep_sec: int = 5):
    """
    使用外部传入的共享 driver：
    1. 打开列表页
    2. 保存调试 HTML
    3. 提取职位详情链接
    """
    driver.get(url)
    time.sleep(sleep_sec)

    print("current_url:", driver.current_url)
    print("title:", driver.title)

    if is_security_verification_page(driver):
        print("[BLOCKED] 当前不是职位列表页，而是安全验证页")
        return []

    html = driver.page_source
    save_debug_html(html, DEBUG_LIST_HTML_PATH)

    soup = BeautifulSoup(html, "html.parser")
    text_preview = soup.get_text(" ", strip=True)[:300]

    print("页面前300个字符：")
    print(text_preview)

    job_links = extract_job_links(soup)

    print(f"\n提取到 {len(job_links)} 条职位详情链接")
    print("前5条链接：")
    for link in job_links[:5]:
        print(link)

    return job_links


def collect_and_save_batch(
    driver,
    list_url: str,
    keyword: str,
    max_jobs: int = 3,
    page_sleep_sec: int = 5,
    detail_sleep_sec: float = 1.5,
):
    """
    批量采集并按 D3 策略落盘：
    - raw_jobs.csv 按 job_id 去重
    - search_hits.csv 记录所有命中关系

    注意：
    - 不在函数内部创建 driver
    - 统一复用外部传入的共享 driver
    """
    print("\n========== 第一步：从列表页提取链接 ==========")
    job_links = open_list_page_and_get_links(driver, list_url, sleep_sec=page_sleep_sec)

    if not job_links:
        print("[ERROR] 没有提取到任何职位详情链接")
        return []

    selected_links = job_links[:max_jobs]
    print(f"\n本次准备采集并落盘 {len(selected_links)} 条链接")

    records = []
    success_count = 0
    fail_count = 0

    for i, link in enumerate(selected_links, start=1):
        print(f"\n========== 第二步：解析并落盘第 {i} 条 ==========")
        print("link:", link)

        try:
            time.sleep(detail_sleep_sec)

            # 关键：把共享 driver 传入详情页解析函数
            record = parse_job_detail(driver=driver, link=link, keyword=keyword)

            records.append(record)
            success_count += 1

            save_record_with_d3_strategy(record)

            print("已完成：", record["job_title"], "|", record["company_name"], "|", record["job_id"])

        except Exception as e:
            fail_count += 1
            print(f"[ERROR] 第 {i} 条处理失败: {e}")
            continue

    print("\n========== 批量采集结束 ==========")
    print(f"成功: {success_count} 条")
    print(f"失败: {fail_count} 条")

    return records


def collect_single_keyword_page_range(
    driver,
    keyword: str,
    base_list_url: str,
    start_page: int = 1,
    end_page: int = 2,
    max_jobs_per_page: int = 20,
):
    """
    单关键词、指定页码范围采集：
    - 从 p1 基础 URL 自动生成 p{start_page} ... p{end_page}
    - 每页解析前 max_jobs_per_page 条职位
    - 使用 D3 策略落盘
    - 全程复用同一个 driver
    """
    all_records = []

    print("\n==============================")
    print(f"开始采集关键词：{keyword}")
    print(f"基础 URL: {base_list_url}")
    print(f"页码范围: {start_page} - {end_page}")
    print(f"每页前 {max_jobs_per_page} 条")
    print("==============================")

    for page_num in range(start_page, end_page + 1):
        page_url = build_paged_url(base_list_url, page_num)

        print(f"\n---------- 处理第 {page_num} 页 ----------")
        print("page_url:", page_url)

        try:
            records = collect_and_save_batch(
                driver=driver,
                list_url=page_url,
                keyword=keyword,
                max_jobs=max_jobs_per_page,
            )
            all_records.extend(records)

        except Exception as e:
            print(f"[ERROR] 第 {page_num} 页处理失败: {e}")
            continue

    print("\n========== 单关键词区间采集结束 ==========")
    print(f"累计返回记录数：{len(all_records)}")

    return all_records


def collect_multiple_keywords_page_range(
    keyword_configs,
    start_page: int = 1,
    end_page: int = 2,
    max_jobs_per_page: int = 20,
    headless: bool = False
):
    """
    多关键词、指定页码范围采集：
    - 整个批次只创建 1 个 driver
    - 每个关键词使用同一组 start_page ~ end_page
    - 每页解析前 max_jobs_per_page 条职位
    """
    all_records = []
    driver = None

    try:
        driver = build_driver(headless=headless)

        for i, config in enumerate(keyword_configs, start=1):
            keyword = config["keyword"]
            base_list_url = config["list_url"]

            print(f"\n==============================")
            print(f"开始采集第 {i} 个关键词：{keyword}")
            print(f"页码范围: {start_page}-{end_page}")
            print("==============================")

            try:
                records = collect_single_keyword_page_range(
                    driver=driver,
                    keyword=keyword,
                    base_list_url=base_list_url,
                    start_page=start_page,
                    end_page=end_page,
                    max_jobs_per_page=max_jobs_per_page,
                )
                all_records.extend(records)

            except Exception as e:
                print(f"[ERROR] 关键词 {keyword} 采集失败: {e}")
                continue

    finally:
        safe_quit_driver(driver)

    print("\n========== 多关键词区间采集结束 ==========")
    print(f"累计返回记录数：{len(all_records)}")

    return all_records


if __name__ == "__main__":
    KEYWORD_LIST_CONFIG = [
        {
            "keyword": "数据分析师",
            "list_url": "https://m.zhaopin.com/sou/kwCLO66RII0PJP0NG8/p1",
            "priority": "core"
        },
        {
            "keyword": "商业分析师",
            "list_url": "https://m.zhaopin.com/sou/kwAL34S6II0PJP0NG8/p1",
            "priority": "core"
        },
        {
            "keyword": "运营分析师",
            "list_url": "https://m.zhaopin.com/sou/kwHV8889AI0PJP0NG8/p1",
            "priority": "core"
        },
        {
            "keyword": "BI分析师",
            "list_url": "https://m.zhaopin.com/sou/kw01100IAI0PJP0NG8/p1",
            "priority": "core"
        },
        {
            "keyword": "数据分析实习生",
            "list_url": "https://m.zhaopin.com/sou/kwCLO66RII0PJP0MSU9PG7A7O/p1",
            "priority": "core"
        },
        {
            "keyword": "数据运营分析师",
            "list_url": "https://m.zhaopin.com/sou/kwCLO66RKFQ222AKG6CU85S20/p1",
            "priority": "extend"
        },
        {
            "keyword": "商业智能分析师",
            "list_url": "https://m.zhaopin.com/sou/kwAL34S6J6FA0FQKG6CU85S20/p1",
            "priority": "extend"
        },
        {
            "keyword": "用户分析师",
            "list_url": "https://m.zhaopin.com/sou/kwEKK64DQI0PJP0NG8/p1",
            "priority": "extend"
        },
        {
            "keyword": "经营分析师",
            "list_url": "https://m.zhaopin.com/sou/kwFR7O89AI0PJP0NG8/p1",
            "priority": "extend"
        },
        {
            "keyword": "商务分析师",
            "list_url": "https://m.zhaopin.com/sou/kwAL3558AI0PJP0NG8/p1",
            "priority": "extend"
        },
    ]

    records = collect_multiple_keywords_page_range(
        keyword_configs=KEYWORD_LIST_CONFIG,
        start_page=25,
        end_page=28,
        max_jobs_per_page=20,
        headless=False
    )

    print("\n========== 最终记录预览 ==========")
    print(f"共返回 {len(records)} 条记录")
    for r in records[:30]:
        print(r["keyword"], "|", r["job_title"], "|", r["company_name"], "|", r["job_id"])