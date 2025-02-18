from rss_feed_saver import RSSFeedSaver  # Import the RSSFeedSaver class
from rss_feed_parser import RSSFeedParser
import sys
import os
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..')))


if __name__ == "__main__":
    # Define your RSS feed URLs
    RSS_FEED_URLS = [
        'http://www.dn.se/nyheter/m/rss/',
        'https://rss.aftonbladet.se/rss2/small/pages/sections/senastenytt/',
        'https://feeds.expressen.se/nyheter/',
        'http://www.svd.se/?service=rss',
        'http://api.sr.se/api/rss/program/83?format=145',
        'http://www.svt.se/nyheter/rss.xml'
    ]

    # Create parser and saver
    parser = RSSFeedParser(RSS_FEED_URLS)
    saver = RSSFeedSaver(parser)

    # Save feeds
    filepath, count = saver.save_feeds()

    # Print summary
    print(f"Successfully saved {count} feed entries to {filepath}")
