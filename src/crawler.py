"""
SHL Product Catalogue Crawler
Phase 1: Extract all Individual Test Solutions from SHL's public catalogue

Target: https://www.shl.com/solutions/products/product-catalog/
Focus: Individual Test Solutions ONLY (type=1)
Ignore: Pre-packaged Job Solutions (type=2)
Minimum expected: 377 tests
"""

import json
import time
import re
import hashlib
from pathlib import Path
from urllib.parse import urljoin, urlparse
from typing import Optional

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


class SHLCrawler:
    """Crawler for SHL Product Catalogue - Individual Test Solutions"""
    
    BASE_URL = "https://www.shl.com/solutions/products/product-catalog/"
    INDIVIDUAL_TESTS_URL = "https://www.shl.com/solutions/products/product-catalog/?start={}&type=1"
    
    # Alternative URL patterns observed
    ALT_BASE_URL = "https://www.shl.com/products/product-catalog/"
    ALT_INDIVIDUAL_TESTS_URL = "https://www.shl.com/products/product-catalog/?start={}&type=1"
    
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Connection": "keep-alive",
    }
    
    PAGE_SIZE = 12  # Items per page
    REQUEST_DELAY = 1.0  # Seconds between requests (be polite)
    
    def __init__(self, output_dir: str = "data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)
        self.tests = []
        self.seen_urls = set()
        
    def _get_page(self, url: str, retries: int = 3) -> Optional[BeautifulSoup]:
        """Fetch a page with retry logic"""
        for attempt in range(retries):
            try:
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                return BeautifulSoup(response.content, "lxml")
            except requests.RequestException as e:
                print(f"  [WARN] Attempt {attempt + 1} failed for {url}: {e}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
        return None
    
    def _extract_test_type_codes(self, cell_text: str) -> list[str]:
        """Extract test type codes from a cell (e.g., 'A B C K P')"""
        # Test types: A=Ability, B=Behavior, C=Cognitive, D=Development, E=Emotional, K=Knowledge, P=Personality, S=Simulation
        valid_codes = {"A", "B", "C", "D", "E", "K", "P", "S"}
        codes = []
        for char in cell_text.upper():
            if char in valid_codes:
                codes.append(char)
        return codes
    
    def _parse_test_row(self, row, table_type: str) -> Optional[dict]:
        """Parse a single table row into test data"""
        cells = row.find_all("td")
        if len(cells) < 4:
            return None
        
        # First cell contains the test name and link
        name_cell = cells[0]
        link = name_cell.find("a")
        
        if not link:
            return None
        
        test_name = link.get_text(strip=True)
        test_url = link.get("href", "")
        
        # Normalize URL
        if test_url and not test_url.startswith("http"):
            test_url = urljoin("https://www.shl.com", test_url)
        
        # Skip if already seen (duplicate check)
        if test_url in self.seen_urls:
            return None
        
        # Extract other columns
        remote_testing = "Yes" if cells[1].find("span", class_="catalogue__circle") else "No"
        adaptive_irt = "Yes" if cells[2].find("span", class_="catalogue__circle") else "No"
        test_type_codes = self._extract_test_type_codes(cells[3].get_text())
        
        # Generate unique ID based on URL
        test_id = hashlib.md5(test_url.encode()).hexdigest()[:12]
        
        return {
            "test_id": test_id,
            "test_name": test_name,
            "url": test_url,
            "remote_testing": remote_testing,
            "adaptive_irt": adaptive_irt,
            "test_type_codes": test_type_codes,
            "category": table_type,
        }
    
    def _get_total_pages(self, soup: BeautifulSoup) -> int:
        """Extract total number of pages from pagination"""
        # Look for pagination links
        pagination = soup.find("nav", class_="pagination") or soup.find("div", class_="pagination")
        if pagination:
            # Find the last page number
            page_links = pagination.find_all("a")
            max_page = 1
            for link in page_links:
                text = link.get_text(strip=True)
                if text.isdigit():
                    max_page = max(max_page, int(text))
            return max_page
        
        # Alternative: look for page links in any format
        links = soup.find_all("a", href=re.compile(r"start=\d+.*type=1"))
        max_start = 0
        for link in links:
            href = link.get("href", "")
            match = re.search(r"start=(\d+)", href)
            if match:
                max_start = max(max_start, int(match.group(1)))
        
        if max_start > 0:
            return (max_start // self.PAGE_SIZE) + 1
        
        return 1
    
    def _extract_tests_from_page(self, soup: BeautifulSoup) -> list[dict]:
        """Extract all Individual Test Solutions from a page"""
        tests = []
        
        # Find all tables on the page
        tables = soup.find_all("table")
        
        for table in tables:
            # Check if this is the Individual Test Solutions table
            # Look for the header row
            header = table.find("thead") or table.find("tr")
            if header:
                header_text = header.get_text()
                # Skip Pre-packaged Job Solutions table
                if "Pre-packaged" in header_text:
                    continue
                if "Individual Test Solutions" in header_text or "Individual" in header_text:
                    table_type = "Individual Test Solutions"
                else:
                    # Default to Individual if we can't determine
                    table_type = "Individual Test Solutions"
            else:
                table_type = "Individual Test Solutions"
            
            # Find all data rows
            tbody = table.find("tbody")
            rows = tbody.find_all("tr") if tbody else table.find_all("tr")[1:]  # Skip header
            
            for row in rows:
                test = self._parse_test_row(row, table_type)
                if test:
                    self.seen_urls.add(test["url"])
                    tests.append(test)
        
        return tests
    
    def _fetch_test_details(self, test: dict) -> dict:
        """Fetch detailed information from individual test page"""
        url = test["url"]
        soup = self._get_page(url)
        
        if not soup:
            test["description"] = ""
            test["skills_measured"] = []
            test["duration"] = ""
            return test
        
        # Extract description
        description_elem = soup.find("div", class_="product-catalogue-training__body") or \
                          soup.find("div", class_="content") or \
                          soup.find("article")
        
        if description_elem:
            # Get paragraph text
            paragraphs = description_elem.find_all("p")
            description = " ".join(p.get_text(strip=True) for p in paragraphs[:3])
        else:
            description = ""
        
        test["description"] = description[:1000]  # Limit length
        
        # Extract skills/competencies measured
        skills = []
        skills_section = soup.find(string=re.compile(r"(Skills|Competencies|Measures)", re.I))
        if skills_section:
            parent = skills_section.find_parent()
            if parent:
                skills_list = parent.find_next("ul")
                if skills_list:
                    skills = [li.get_text(strip=True) for li in skills_list.find_all("li")]
        
        test["skills_measured"] = skills[:10]  # Limit
        
        # Extract duration if available
        duration_elem = soup.find(string=re.compile(r"\d+\s*(min|minute)", re.I))
        test["duration"] = duration_elem.strip() if duration_elem else ""
        
        return test
    
    def crawl_catalogue(self, fetch_details: bool = True) -> list[dict]:
        """
        Main crawling method - extracts all Individual Test Solutions
        
        Args:
            fetch_details: Whether to fetch detailed info from each test page
            
        Returns:
            List of test dictionaries
        """
        print("=" * 60)
        print("SHL Catalogue Crawler - Phase 1")
        print("=" * 60)
        print(f"Target: Individual Test Solutions")
        print(f"Base URL: {self.BASE_URL}")
        print()
        
        # First, get the main page to determine total pages
        print("[1/4] Fetching main catalogue page...")
        main_soup = self._get_page(self.BASE_URL)
        
        if not main_soup:
            # Try alternative URL
            print("  [INFO] Trying alternative URL...")
            main_soup = self._get_page(self.ALT_BASE_URL)
            
        if not main_soup:
            raise RuntimeError("Failed to fetch catalogue page")
        
        total_pages = self._get_total_pages(main_soup)
        print(f"  Found {total_pages} pages of Individual Test Solutions")
        
        # Extract from first page
        initial_tests = self._extract_tests_from_page(main_soup)
        self.tests.extend(initial_tests)
        print(f"  Extracted {len(initial_tests)} tests from page 1")
        
        # Crawl all pagination pages
        print(f"\n[2/4] Crawling {total_pages - 1} additional pages...")
        
        for page in tqdm(range(1, total_pages), desc="Crawling pages"):
            time.sleep(self.REQUEST_DELAY)
            
            start = page * self.PAGE_SIZE
            url = self.INDIVIDUAL_TESTS_URL.format(start)
            
            soup = self._get_page(url)
            if not soup:
                # Try alternative URL
                url = self.ALT_INDIVIDUAL_TESTS_URL.format(start)
                soup = self._get_page(url)
            
            if soup:
                page_tests = self._extract_tests_from_page(soup)
                self.tests.extend(page_tests)
        
        print(f"\n  Total tests from listings: {len(self.tests)}")
        
        # Fetch detailed information for each test
        if fetch_details:
            print(f"\n[3/4] Fetching detailed info for {len(self.tests)} tests...")
            for i, test in enumerate(tqdm(self.tests, desc="Fetching details")):
                time.sleep(self.REQUEST_DELAY * 0.5)  # Slightly faster for detail pages
                self._fetch_test_details(test)
        
        # Validation
        print(f"\n[4/4] Validation...")
        print(f"  Total tests extracted: {len(self.tests)}")
        print(f"  Unique URLs: {len(self.seen_urls)}")
        
        # Check for minimum count
        if len(self.tests) < 377:
            print(f"  [WARN] Expected >= 377 tests, got {len(self.tests)}")
        else:
            print(f"  [OK] Meets minimum requirement (>= 377)")
        
        return self.tests
    
    def save_catalogue(self, filename: str = "catalog_raw.json"):
        """Save extracted catalogue to JSON file"""
        output_path = self.output_dir / filename
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({
                "metadata": {
                    "source": self.BASE_URL,
                    "crawl_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "total_tests": len(self.tests),
                    "category": "Individual Test Solutions"
                },
                "tests": self.tests
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n[SAVED] Catalogue saved to: {output_path}")
        print(f"        Total tests: {len(self.tests)}")
        
        return output_path


def main():
    """Main entry point for crawling"""
    crawler = SHLCrawler(output_dir="data")
    
    try:
        tests = crawler.crawl_catalogue(fetch_details=True)
        output_path = crawler.save_catalogue()
        
        print("\n" + "=" * 60)
        print("PHASE 1 COMPLETE")
        print("=" * 60)
        print(f"Total tests extracted: {len(tests)}")
        print(f"Output file: {output_path}")
        
        # Validation summary
        if len(tests) >= 377:
            print("\n✓ VALIDATION PASSED: >= 377 Individual Test Solutions")
        else:
            print(f"\n✗ VALIDATION FAILED: Only {len(tests)} tests (expected >= 377)")
            print("  Manual review required before proceeding to Phase 2")
            
    except Exception as e:
        print(f"\n[ERROR] Crawling failed: {e}")
        raise


if __name__ == "__main__":
    main()
