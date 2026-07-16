"""
爬虫
1. 美股：从 SEC EDGAR 下载 10-K 年报
2. A股：自动获取 orgId 并从巨潮资讯下载年报 PDF
3. 通用：爬取网页内容
"""

import os
import time
import hashlib
import requests
import json
import urllib3
from pathlib import Path
from typing import List, Dict, Optional

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class SECEdgarDownloader:
    """SEC EDGAR 文档下载器 (美股)"""
    
    def __init__(self, output_dir="data/raw/sec", company_name="FinRAG_User", email="admin@example.com"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.user_agent = f"{company_name} {email}"

    def download_10k_filings(self, ticker: str, amount: int = 3) -> List[Path]:
        """下载 10-K 年报"""
        try:
            from sec_edgar_downloader import Downloader
            dl = Downloader(self.user_agent.split()[0], self.user_agent.split()[1], self.output_dir)
            
            print(f"[SEC] 正在下载 {ticker} 最近 {amount} 年的 10-K 年报...")
            dl.get("10-K", ticker, limit=amount)
            
            target_dir = self.output_dir / "sec-edgar-filings" / ticker / "10-K"
            downloaded_files = []
            if target_dir.exists():
                downloaded_files.extend(list(target_dir.rglob("*.html")))
                downloaded_files.extend(list(target_dir.rglob("*.txt")))
            
            print(f"[SEC] {ticker} 下载完成，获取 {len(downloaded_files)} 个文件")
            return downloaded_files
        except Exception as e:
            print(f"[SEC] 下载出错: {e}")
            return []


class CNINFOCrawler:
    """巨潮资讯网爬虫 (增强版 - 自动解析 OrgID)"""
    
    def __init__(self, output_dir="data/raw/cninfo"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # API 接口
        self.search_api = "http://www.cninfo.com.cn/new/information/topSearch/query"
        self.query_api = "http://www.cninfo.com.cn/new/hisAnnouncement/query"
        self.base_url = "http://static.cninfo.com.cn/"
    
    def _get_org_id(self, code: str) -> Optional[Dict]:
        """关键修复：通过代码获取 orgId 和板块信息"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8'
            }
            # 搜索股票
            resp = requests.post(self.search_api, data={'keyWord': code, 'maxNum': 10}, headers=headers, timeout=5)
            if resp.status_code == 200:
                results = resp.json()
                for item in results:
                    # 匹配股票代码
                    if item.get('code') == code:
                        return {
                            'orgId': item['orgId'],
                            'category': item.get('category', 'A股'), # A股/港股/三板等
                            'plate': 'sse' if item.get('orgId').startswith('gssh') else 'szse' # 简单判断沪深
                        }
            return None
        except Exception as e:
            print(f"[CNINFO] OrgID 解析失败: {e}")
            return None

    def fetch_announcements_meta(self, symbol: str, limit: int = 5) -> List[Dict]:
        """获取年报下载链接"""
        print(f"[CNINFO] 正在解析股票 {symbol} 信息...")
        
        # 1. 获取 OrgID (API 必填参数)
        stock_info = self._get_org_id(symbol)
        if not stock_info:
            print(f"未找到股票 {symbol} 的 OrgID，请检查代码是否正确")
            return []
            
        org_id = stock_info['orgId']
        column = 'sse' if symbol.startswith('6') or stock_info['plate'] == 'sse' else 'szse'
        
        print(f"识别成功: OrgID={org_id}, 板块={column}")
        
        try:
            # 2. 构造查询参数
            params = {
                'pageNum': 1,
                'pageSize': limit,
                'column': column,
                'tabName': 'fulltext',
                'plate': '',
                'stock': f"{symbol},{org_id}", # 核心修复：格式必须是 "代码,orgId"
                'searchkey': '',
                'secid': '',
                'category': 'category_ndbg_szsh', # 指定只下载年报
                'trade': '',
                'seDate': '',
                'sortName': '',
                'sortType': '',
                'isHLtitle': 'true'
            }
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'X-Requested-With': 'XMLHttpRequest',
                'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8'
            }

            resp = requests.post(self.query_api, data=params, headers=headers, timeout=10)
            data = resp.json()
            
            if not data.get('announcements'):
                print(f"{symbol} 暂无符合条件的年报")
                return []
            
            results = []
            for item in data['announcements']:
                title = item['announcementTitle']
                if "摘要" in title: continue # 跳过摘要
                
                if item.get('adjunctUrl'):
                    results.append({
                        "title": title,
                        "url": self.base_url + item['adjunctUrl'],
                        "date": item.get('announcementTime')
                    })
            
            final_res = results[:limit]
            print(f"成功获取 {len(final_res)} 份年报元数据")
            return final_res
            
        except Exception as e:
            print(f"[CNINFO] API 查询异常: {e}")
            return []

    def download_pdf(self, url: str, title: str, symbol: str) -> Optional[Path]:
        """下载 PDF"""
        try:
            safe_title = "".join([c for c in title if c.isalnum() or c in (' ', '-', '_', '.')]).strip()
            if len(safe_title) > 50: safe_title = safe_title[:50]
            filename = f"{symbol}_{safe_title}.pdf"
            filepath = self.output_dir / filename
            
            if filepath.exists():
                print(f"[CNINFO] 已存在: {filename}")
                return filepath
                
            print(f"下载中: {filename} ...")
            headers = {'User-Agent': 'Mozilla/5.0'}
            # verify=False 规避 SSL 报错, timeout=60 防止大文件超时
            resp = requests.get(url, headers=headers, stream=True, verify=False, timeout=60)
            
            if resp.status_code == 200:
                with open(filepath, 'wb') as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)
                print("下载完成")
                return filepath
            else:
                print(f"下载失败 HTTP {resp.status_code}")
                return None
        except Exception as e:
            print(f"下载异常: {e}")
            return None


class GeneralCrawler:
    """通用网页爬虫"""
    
    def __init__(self, output_dir="data/raw/web"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def crawl(self, url: str) -> Optional[Path]:
        print(f"[WEB] 正在爬取: {url}")
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            url_hash = hashlib.md5(url.encode('utf-8')).hexdigest()[:8]
            filename = f"web_{url_hash}.html"
            filepath = self.output_dir / filename
            
            with open(filepath, 'wb') as f:
                f.write(response.content)
            print(f"[WEB] 保存成功: {filename}")
            return filepath
        except Exception as e:
            print(f"[WEB] 爬取失败 {url}: {e}")
            return None


class FinancialCrawlerFacade:
    """统一运行入口"""
    
    def __init__(self, base_dir="my_rag_data"):
        self.base_dir = Path(base_dir)
        self.sec = SECEdgarDownloader(output_dir=self.base_dir / "sec")
        self.cninfo = CNINFOCrawler(output_dir=self.base_dir / "cninfo")
        self.web = GeneralCrawler(output_dir=self.base_dir / "web")
    
    def run_pipeline(self, sec_tickers=[], cn_symbols=[], urls=[]):
        print(f"任务开始! 数据目录: {self.base_dir.absolute()}")
        print("-" * 50)
        
        # 1. 美股
        for ticker in sec_tickers:
            self.sec.download_10k_filings(ticker)
            
        # 2. A股
        for symbol in cn_symbols:
            metas = self.cninfo.fetch_announcements_meta(symbol, limit=2)
            for meta in metas:
                self.cninfo.download_pdf(meta['url'], meta['title'], symbol)
                time.sleep(1)
        
        # 3. 网页
        for url in urls:
            self.web.crawl(url)
            
        print("-" * 50)
        print("任务全部完成")


if __name__ == "__main__":
    crawler = FinancialCrawlerFacade()
    
    us_stocks = ["AAPL"]
    
    # A股 (000001:平安银行, 600519:贵州茅台)
    cn_stocks = ["000001", "600519"] 
    
    # 网页 (改为百度测试)
    web_urls = ["https://www.baidu.com"]
    
    crawler.run_pipeline(
        sec_tickers=us_stocks,
        cn_symbols=cn_stocks,
        urls=web_urls
    )