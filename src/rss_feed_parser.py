import feedparser
import json
from datetime import datetime
import email.utils
import urllib.parse


class RSSFeedParser:
    def __init__(self, feed_urls):
        self.feed_urls = feed_urls

    def fetch_and_parse_feeds(self):
        """Fetch and parse RSS feeds"""
        posts = []

        for url in self.feed_urls:
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries:
                    published_raw = entry.get('published', '')
                    # Remove the trailing comma if published_raw is a tuple
                    if isinstance(published_raw, tuple):
                        published_raw = published_raw[0]

                    published_formatted = ""
                    # Try the email.utils parser first (handles RFC 2822 format with timezone)
                    try:
                        parsed_time_tuple = email.utils.parsedate_tz(
                            published_raw)
                        if parsed_time_tuple:
                            # Convert time tuple to UTC timestamp
                            timestamp = email.utils.mktime_tz(
                                parsed_time_tuple)
                            # Convert timestamp to datetime object
                            dt = datetime.fromtimestamp(timestamp)
                            published_formatted = dt.strftime(
                                "%Y-%m-%d %H:%M:%S")
                    except:
                        # Fallback to previous method if email.utils parser fails
                        possible_formats = [
                            "%a, %d %b %Y %H:%M:%S %z",  # Format with timezone offset
                            "%a, %d %b %Y %H:%M:%S %Z",
                            "%Y-%m-%dT%H:%M:%SZ",
                            "%Y-%m-%dT%H:%M:%S.%fZ",
                            "%Y-%m-%d %H:%M:%S"
                        ]

                        for fmt in possible_formats:
                            try:
                                parsed_date = datetime.strptime(
                                    published_raw, fmt)
                                published_formatted = parsed_date.strftime(
                                    "%Y-%m-%d %H:%M:%S")
                                break
                            except ValueError:
                                continue

                    link = entry.get('link', 'No link')
                    post = {
                        'title': entry.get('title', 'No title'),
                        'link': link,
                        'summary': entry.get('summary', 'No summary'),
                        "published": published_formatted,
                        "source_news": self.get_news_outlet(link)
                    }
                    posts.append(post)
            except Exception as e:
                print(f"Failed to parse feed from {url}: {e}")
        return posts

    @staticmethod
    def get_news_outlet(url):
        """
        Given a URL, return a friendly news outlet name based on its domain.
        """
        try:
            parsed = urllib.parse.urlparse(url)
            domain = parsed.netloc.lower()
        except Exception:
            return "Unknown"
        if "dn.se" in domain:
            return "Dagens Nyheter"
        elif "aftonbladet.se" in domain:
            return "Aftonbladet"
        elif "expressen.se" in domain:
            return "Expressen"
        elif "svd.se" in domain:
            return "SvD"
        elif "sr.se" in domain or "sverigesradio.se" in domain:
            return "Sveriges Radio"
        elif "svt.se" in domain:
            return "SVT"
        else:
            return domain
