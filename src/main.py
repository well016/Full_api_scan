"""
Scan GitHub for available OpenAI API Keys
"""

import argparse
import errno
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

import rich
from selenium import webdriver
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from tqdm import tqdm
from urllib3.exceptions import MaxRetryError, ReadTimeoutError as UrllibReadTimeoutError

from configs import ENV_PROVIDER_VARIABLES, KEYWORDS, LANGUAGES, PATHS, PROVIDER_PATTERNS
from manager import CookieManager, DatabaseManager, ProgressManager
from utils import check_key, detect_provider

FORMAT = "%(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt="[%X]")
log = logging.getLogger("ChatGPT-API-Leakage")
httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)


def _session_lost(exc: BaseException) -> bool:
    """True when Selenium/chromedriver session is gone (crash, kill, port closed)."""
    if isinstance(exc, MaxRetryError):
        return True
    if isinstance(exc, OSError) and getattr(exc, "errno", None) in (
        errno.ECONNREFUSED,
        errno.EPIPE,
        errno.ECONNRESET,
    ):
        return True
    if type(exc).__name__ in ("ConnectionRefusedError", "BrokenPipeError"):
        return True
    msg = str(exc).lower()
    needles = (
        "invalid session id",
        "connection refused",
        "failed to establish a new connection",
        "chrome not reachable",
        "session deleted",
        "disconnected: not connected to devtools",
        "target window already closed",
        "no such window",
        "web view not found",
        "session not created",
    )
    return any(s in msg for s in needles)


class APIKeyLeakageScanner:
    """
    Scan GitHub for available OpenAI API Keys
    """

    def __init__(self, db_file: str, keywords: list, languages: list):
        self.db_file = db_file
        self.driver: webdriver.Chrome | None = None
        self.cookies: CookieManager | None = None
        rich.print(f"📂 Opening database file {self.db_file}")

        self.dbmgr = DatabaseManager(self.db_file)

        self.keywords = keywords
        self.languages = languages
        self.env_extractors: list[tuple[str, str, re.Pattern]] = []
        for provider, variable_names in ENV_PROVIDER_VARIABLES.items():
            for variable in variable_names:
                self.env_extractors.append(
                    (
                        provider,
                        variable,
                        re.compile(
                            rf"\b{re.escape(variable)}\b\s*(?:=|:)\s*(?:['\"])?([^\s'\"`;,#]+)",
                            flags=re.IGNORECASE,
                        ),
                    )
                )
        self.candidate_urls = []
        candidate_url_set: set[str] = set()
        for pattern in PROVIDER_PATTERNS:
            regex = pattern["regex"]
            too_many_results = pattern["have_many_results"]
            # Add the paths to the search query
            for path in PATHS:
                candidate_url_set.add(f"https://github.com/search?q=(/{regex.pattern}/)+AND+({path})&type=code&ref=advsearch")

            for language in self.languages:
                if too_many_results:  # if the regex is too many results, then we need to add AND condition
                    candidate_url_set.add(f"https://github.com/search?q=(/{regex.pattern}/)+language:{language}&type=code&ref=advsearch")
                else:  # if the regex is not too many results, then we just need the regex
                    candidate_url_set.add(f"https://github.com/search?q=(/{regex.pattern}/)&type=code&ref=advsearch")

        # Smart mode: also search by env variable names to catch providers with non-unique token format.
        for variable_names in ENV_PROVIDER_VARIABLES.values():
            for variable in variable_names:
                for path in PATHS:
                    candidate_url_set.add(f"https://github.com/search?q=\"{variable}\"+AND+({path})&type=code&ref=advsearch")
                for language in self.languages:
                    candidate_url_set.add(f"https://github.com/search?q=\"{variable}\"+language:{language}&type=code&ref=advsearch")

        self.candidate_urls = sorted(candidate_url_set)

    @staticmethod
    def _build_chrome_options() -> webdriver.ChromeOptions:
        options = webdriver.ChromeOptions()
        options.add_argument("--ignore-certificate-errors")
        options.add_argument("--ignore-ssl-errors")
        # Do not wait for every asset; DOM-ready is enough for code search pages.
        options.page_load_strategy = "eager"
        return options

    def _restart_driver(self):
        """
        Restart webdriver session and restore GitHub cookies.
        """
        if self.driver is not None:
            try:
                self.driver.quit()
            except Exception:  # pylint: disable=broad-except
                pass

        self.driver = webdriver.Chrome(options=self._build_chrome_options())
        self.driver.implicitly_wait(3)
        self.driver.set_page_load_timeout(120)
        self.cookies = CookieManager(self.driver)

        if os.path.exists("cookies.pkl"):
            # We must open a GitHub page before injecting cookies.
            self.driver.get("https://github.com/login")
            self.cookies.load()
            self.cookies.verify_user_login()

    def _navigation_target_reached(self, url: str) -> bool:
        """True when the browser landed on the requested page (possibly after window.stop())."""
        if self.driver is None:
            return False
        try:
            current = self.driver.current_url
        except WebDriverException:
            return False
        target = urlparse(url)
        current_parsed = urlparse(current)
        if target.netloc and target.netloc != current_parsed.netloc:
            return False
        if target.path and target.path.rstrip("/") in current.rstrip("/"):
            return True
        return url.split("?", 1)[0] in current

    def _try_accept_partial_page_load(self, url: str) -> bool:
        if self.driver is None:
            return False
        try:
            self.driver.execute_script("window.stop();")
        except Exception:  # pylint: disable=broad-except
            pass
        if self._navigation_target_reached(url):
            rich.print(f"🟢 Partial page load accepted for '{url[:80]}'")
            return True
        return False

    def _safe_driver_get(self, url: str, retries: int = 5):
        """
        Navigate with retries to avoid transient Selenium/driver read timeouts.
        Restarts Chrome when the WebDriver session dies (connection refused, invalid session, etc.).
        """
        if self.driver is None:
            raise ValueError("Driver is not initialized")

        navigation_errors = (TimeoutException, WebDriverException, UrllibReadTimeoutError, MaxRetryError)

        for attempt in range(1, retries + 1):
            try:
                self.driver.get(url)
                return
            except navigation_errors as exc:
                rich.print(f"🟡 Navigation retry {attempt}/{retries} for '{url[:80]}' due to: {type(exc).__name__}")
                if _session_lost(exc):
                    rich.print("🔁 WebDriver session lost — restarting Chrome …")
                    self._restart_driver()
                elif isinstance(exc, TimeoutException) and self._try_accept_partial_page_load(url):
                    return
                elif attempt < retries:
                    try:
                        self.driver.execute_script("window.stop();")
                    except Exception:  # pylint: disable=broad-except
                        pass
                if attempt < retries:
                    time.sleep(min(2 * attempt, 8))
                    continue
                rich.print("🔁 Navigation failed — restarting Chrome before giving up …")
                self._restart_driver()
                raise
            except Exception as exc:  # pylint: disable=broad-except
                if _session_lost(exc):
                    rich.print(f"🟡 Navigation retry {attempt}/{retries} ({type(exc).__name__}), restarting Chrome …")
                    self._restart_driver()
                    if attempt < retries:
                        time.sleep(min(2 * attempt, 8))
                        continue
                raise

    def _wait_expand_page_ready(self, timeout: float = 20.0) -> None:
        """Wait for expanded code page instead of a fixed sleep when possible."""
        if self.driver is None:
            raise ValueError("Driver is not initialized")
        try:
            WebDriverWait(self.driver, timeout).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "main, .blob-wrapper, #read-only-cursor-text-area, table.js-file-line-container"))
            )
        except TimeoutException:
            time.sleep(2)

    def _page_source_safe(self, expand_url: str, retries: int = 2) -> str:
        """Return page_source, restarting the browser if the session died."""
        if self.driver is None:
            raise ValueError("Driver is not initialized")
        for attempt in range(retries):
            try:
                return self.driver.page_source
            except Exception as exc:  # pylint: disable=broad-except
                if not _session_lost(exc):
                    raise
                if attempt >= retries - 1:
                    raise
                rich.print(f"🔁 Lost session while reading page ({type(exc).__name__}), restarting …")
                self._restart_driver()
                self._safe_driver_get(expand_url)
                self._wait_expand_page_ready()
        raise RuntimeError("_page_source_safe: unreachable")  # pragma: no cover

    def login_to_github(self):
        """
        Login to GitHub
        """
        rich.print("🌍 Opening Chrome ...")

        self.driver = webdriver.Chrome(options=self._build_chrome_options())
        self.driver.implicitly_wait(3)
        self.driver.set_page_load_timeout(120)

        self.cookies = CookieManager(self.driver)

        cookie_exists = os.path.exists("cookies.pkl")
        self._safe_driver_get("https://github.com/login")

        if not cookie_exists:
            rich.print("🤗 No cookies found, please login to GitHub first")
            input("Press Enter after you logged in: ")
            self.cookies.save()
        else:
            rich.print("🍪 Cookies found, loading cookies")
            self.cookies.load()

        self.cookies.verify_user_login()

    def _expand_all_code(self):
        """
        Expand all the code in the current page
        """
        elements = self.driver.find_elements(by=By.XPATH, value="//*[contains(text(), 'more match')]")
        for element in elements:
            element.click()

    @staticmethod
    def _normalize_env_secret(raw_value: str) -> str:
        value = raw_value.strip()
        while value and value[-1] in ",;)}]":
            value = value[:-1]
        return value

    @staticmethod
    def _is_placeholder_secret(value: str) -> bool:
        lowered = value.lower()
        if len(value) < 12:
            return True
        if value.startswith(("$", "{", "<")):
            return True
        placeholder_tokens = [
            "your_",
            "your-",
            "example",
            "placeholder",
            "changeme",
            "token_here",
            "api_key_here",
            "replace_me",
            "xxxxx",
            "dummy",
        ]
        return any(token in lowered for token in placeholder_tokens)

    def _extract_env_key_pairs(self, text: str) -> list[tuple[str, str]]:
        extracted: set[tuple[str, str]] = set()
        for provider, _, extractor in self.env_extractors:
            for match in extractor.findall(text):
                candidate = self._normalize_env_secret(match)
                if self._is_placeholder_secret(candidate):
                    continue
                extracted.add((candidate, provider))
        return list(extracted)

    def _extract_pattern_key_pairs(self, text: str, include_too_long: bool = True) -> list[tuple[str, str]]:
        extracted: set[tuple[str, str]] = set()
        for pattern in PROVIDER_PATTERNS:
            too_long = pattern["result_too_long"]
            if too_long and not include_too_long:
                continue
            regex = pattern["regex"]
            provider = pattern["provider"]
            for key in regex.findall(text):
                extracted.add((key, provider))
        return list(extracted)

    def _extract_all_key_pairs(self, text: str, include_too_long: bool = True) -> list[tuple[str, str]]:
        return list(set(self._extract_pattern_key_pairs(text, include_too_long) + self._extract_env_key_pairs(text)))

    def _find_urls_and_apis(self) -> tuple[list[tuple[str, str]], list[str]]:
        """
        Find all the urls and apis in the current page
        """
        apis_found: list[tuple[str, str]] = []
        urls_need_expand = []

        codes = self.driver.find_elements(by=By.CLASS_NAME, value="code-list")  # type: ignore
        for element in codes:
            apis = self._extract_all_key_pairs(element.text, include_too_long=False)

            if len(apis) == 0:
                # Need to show full code. (because the api key is too long)
                # get the <a> tag
                a_tag = element.find_element(by=By.XPATH, value=".//a")
                urls_need_expand.append(a_tag.get_attribute("href"))
            apis_found.extend(apis)

        return list(set(apis_found)), urls_need_expand

    def _process_url(self, url: str):
        """
        Process a search query url
        """
        if self.driver is None:
            raise ValueError("Driver is not initialized")

        self._safe_driver_get(url)

        apis_found: list[tuple[str, str]] = []
        urls_need_expand: list[str] = []

        while True:  # Loop until all the pages are processed
            try:
                # If current webpage is reached the rate limit, then wait for 30 seconds
                if self.driver.find_elements(by=By.XPATH, value="//*[contains(text(), 'You have exceeded a secondary rate limit')]"):
                    for _ in tqdm(range(30), desc="⏳ Rate limit reached, waiting ..."):
                        time.sleep(1)
                    self.driver.refresh()
                    continue

                self._expand_all_code()

                page_apis, page_expand = self._find_urls_and_apis()
                apis_found.extend(page_apis)
                urls_need_expand.extend(page_expand)
                rich.print(f"    🌕 There are {len(page_expand)} urls waiting to be expanded on this page")

                try:
                    next_buttons = self.driver.find_elements(by=By.XPATH, value="//a[@aria-label='Next Page']")
                    rich.print("🔍 Clicking next page")
                    WebDriverWait(self.driver, 5).until(
                        EC.presence_of_element_located((By.XPATH, "//a[@aria-label='Next Page']"))
                    )
                    next_buttons = self.driver.find_elements(by=By.XPATH, value="//a[@aria-label='Next Page']")
                    next_buttons[0].click()
                except Exception as exc:  # pylint: disable=broad-except
                    if _session_lost(exc):
                        rich.print("🔁 Session lost during pagination — restarting Chrome and reloading search …")
                        self._restart_driver()
                        self._safe_driver_get(url)
                        continue
                    rich.print("⚪️ No more pages")
                    break
            except Exception as exc:  # pylint: disable=broad-except
                if _session_lost(exc):
                    rich.print(f"🔁 Session lost while scanning results ({type(exc).__name__}) — restarting …")
                    self._restart_driver()
                    self._safe_driver_get(url)
                    continue
                raise

        urls_need_expand = list(dict.fromkeys(urls_need_expand))

        # Handle the expand_urls
        expand_navigation_errors = (TimeoutException, WebDriverException, UrllibReadTimeoutError, MaxRetryError)
        for u in tqdm(urls_need_expand, desc="🔍 Expanding URLs ..."):
            if self.driver is None:
                raise ValueError("Driver is not initialized")

            with self.dbmgr as mgr:
                if mgr.get_url(u):
                    rich.print(f"    🔑 skipping url '{u[:10]}...{u[-10:]}'")
                    continue

            try:
                self.driver.set_page_load_timeout(60)
                self._safe_driver_get(u, retries=3)
            except expand_navigation_errors as exc:
                rich.print(f"    🔴 Skipping expand URL (navigation failed): '{u[:80]}' ({type(exc).__name__})")
                self._restart_driver()
                continue
            finally:
                self.driver.set_page_load_timeout(120)

            self._wait_expand_page_ready()

            retry = 0
            while retry <= 3:
                matches = self._extract_all_key_pairs(self._page_source_safe(u), include_too_long=True)

                if len(matches) == 0:
                    rich.print(f"    ⚪️ No matches found in the expanded page, retrying [{retry}/3]...")
                    retry += 1
                    if retry <= 3:
                        try:
                            self.driver.refresh()
                            self._wait_expand_page_ready()
                        except expand_navigation_errors as exc:
                            rich.print(f"    🔴 Refresh failed for '{u[:80]}' ({type(exc).__name__}), stopping retries")
                            break
                    time.sleep(2)
                    continue

                with self.dbmgr as mgr:
                    new_apis = [(key, provider) for key, provider in matches if not mgr.key_exists(key, provider)]
                    new_apis = list(set(new_apis))
                apis_found.extend(new_apis)
                rich.print(f"    🔬 Found {len(matches)} matches in the expanded page, adding them to the list")
                for key, provider in matches:
                    rich.print(f"        [{provider}] '{key[:10]}...{key[-10:]}'")

                with self.dbmgr as mgr:
                    mgr.insert_url(u)
                break

        self.check_api_keys_and_save(apis_found)

    def check_api_keys_and_save(self, key_provider_pairs: list[tuple[str, str]]):
        """
        Check a list of API keys
        """
        normalized_pairs = {(key, provider or detect_provider(key)) for key, provider in set(key_provider_pairs)}

        with self.dbmgr as mgr:
            unique_pairs = [(key, provider) for key, provider in normalized_pairs if not mgr.key_exists(key, provider)]

        with ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(lambda kp: check_key(kp[0], provider=kp[1]), unique_pairs))
            with self.dbmgr as mgr:
                for idx, result in enumerate(results):
                    key, provider = unique_pairs[idx]
                    mgr.insert(key, result, provider=provider)

    def search(self, from_iter: int | None = None):
        """
        Search for API keys, and save the results to the database
        """
        progress = ProgressManager()
        total = len(self.candidate_urls)
        pbar = tqdm(
            enumerate(self.candidate_urls),
            total=total,
            desc="🔍 Searching ...",
        )
        if from_iter is None:
            from_iter = progress.load(total=total)

        for idx, url in enumerate(self.candidate_urls):
            if idx < from_iter:
                pbar.update()
                time.sleep(0.05)  # let tqdm print the bar
                log.debug("⚪️ Skip %s", url)
                continue
            try:
                self._process_url(url)
            except (TimeoutException, WebDriverException, UrllibReadTimeoutError, MaxRetryError) as exc:
                rich.print(f"🔴 Skipping search query after navigation failure ({type(exc).__name__}): {url[:120]}")
                self._restart_driver()
            progress.save(idx, total)
            log.debug("🔍 Finished %s", url)
            pbar.update()
        pbar.close()

    def deduplication(self):
        """
        Deduplicate the database
        """
        with self.dbmgr as mgr:
            mgr.deduplicate()

    def update_existed_keys(self):
        """
        Update previously checked API keys in the database with their current status
        """
        with self.dbmgr as mgr:
            rich.print("🔄 Updating existed keys")
            keys = mgr.all_keys()
            for key in tqdm(keys, desc="🔄 Updating existed keys ..."):
                result = check_key(key[0], provider=key[1])
                mgr.delete(key[0], provider=key[1])
                mgr.insert(key[0], result, provider=key[1])

    def update_iq_keys(self):
        """
        Update insuffcient quota keys
        """
        with self.dbmgr as mgr:
            rich.print("🔄 Updating insuffcient quota keys")
            keys = mgr.all_iq_keys()
            for key in tqdm(keys, desc="🔄 Updating insuffcient quota keys ..."):
                result = check_key(key[0], provider=key[1])
                mgr.delete(key[0], provider=key[1])
                mgr.insert(key[0], result, provider=key[1])

    def all_available_keys(self) -> list:
        """
        Get all available keys
        """
        with self.dbmgr as mgr:
            return mgr.all_keys()

    def __del__(self):
        if hasattr(self, "driver") and self.driver is not None:
            self.driver.quit()


def main(from_iter: int | None = None, check_existed_keys_only: bool = False, keywords: list | None = None, languages: list | None = None, check_insuffcient_quota: bool = False):
    """
    Main function to scan GitHub for available OpenAI API Keys
    """
    keywords = KEYWORDS.copy() if keywords is None else keywords
    languages = LANGUAGES.copy() if languages is None else languages

    leakage = APIKeyLeakageScanner("github.db", keywords, languages)

    if not check_existed_keys_only:
        leakage.login_to_github()
        leakage.search(from_iter=from_iter)

    if check_insuffcient_quota:
        leakage.update_iq_keys()

    leakage.update_existed_keys()
    leakage.deduplication()
    keys = leakage.all_available_keys()

    rich.print(f"🔑 [bold green]Available keys ({len(keys)}):[/bold green]")
    for key in keys:
        rich.print(f"[bold green]{key[1]}[/bold green]: {key[0]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-iter", type=int, default=None, help="Start from the specific iteration")
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable debug mode, otherwise INFO mode. Default is False (INFO mode)",
    )
    parser.add_argument(
        "-ceko",
        "--check-existed-keys-only",
        action="store_true",
        default=False,
        help="Only check existed keys",
    )
    parser.add_argument(
        "-ciq",
        "--check-insuffcient-quota",
        action="store_true",
        default=False,
        help="Check and update status of the insuffcient quota keys",
    )
    parser.add_argument(
        "-k",
        "--keywords",
        nargs="+",
        default=KEYWORDS,
        help="Keywords to search",
    )
    parser.add_argument(
        "-l",
        "--languages",
        nargs="+",
        default=LANGUAGES,
        help="Languages to search",
    )
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    main(
        from_iter=args.from_iter,
        check_existed_keys_only=args.check_existed_keys_only,
        keywords=args.keywords,
        languages=args.languages,
        check_insuffcient_quota=args.check_insuffcient_quota,
    )
