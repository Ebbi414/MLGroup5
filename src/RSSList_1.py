import feedparser
from datetime import datetime

# URL for the RSS feed
RSS_FEED_URL = [
    'http://www.dn.se/nyheter/m/rss/',
    'https://rss.aftonbladet.se/rss2/small/pages/sections/senastenytt/',
    'https://feeds.expressen.se/nyheter/',
    'http://www.svd.se/?service=rss',
    'http://api.sr.se/api/rss/program/83?format=145',
    'http://www.svt.se/nyheter/rss.xml'
]

# Initialize printdepositlist as a global variable
printdepositlist = []


def fetch_and_parse_feeds():
    """Fetch and parse RSS feeds"""
    posts = []
    possible_formats = [
        "%a, %d %b %Y %H:%M:%S %Z",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%d %H:%M:%S"
    ]

    for url in RSS_FEED_URL:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                published_raw = entry.get("published", "")
                published_formatted = ""
                for fmt in possible_formats:
                    try:
                        parsed_date = datetime.strptime(published_raw, fmt)
                        published_formatted = parsed_date.strftime(
                            "%Y-%m-%d %H:%M:%S")
                        break
                    except ValueError:
                        continue

                post = {
                    "title": entry.get("title", ""),
                    "summary": entry.get("description", ""),
                    "link": entry.get("link", ""),
                    "published": published_formatted
                }
                posts.append(post)
        except Exception as e:
            print(f"Failed to parse feed from {url}: {e}")
    return posts


def OnlyTitlesandSummaries(posts):
    """Extract titles and summaries"""
    only_titles_and_summaries = []
    for x in posts:
        tempdict = {
            "title": x.get("title", ""),
            "summary": x.get("summary", ""),
            "link": x.get("link", ""),
            "published": x.get("published", "")
        }
        only_titles_and_summaries.append(tempdict)
    return only_titles_and_summaries


def TitleAndSummaryList(only_titles_and_summaries):
    """Combine titles and summaries"""
    title_and_summary_list = []
    for item in only_titles_and_summaries:
        combined = item["title"] + " " + item["summary"]
        title_and_summary_list.append([combined])
    return title_and_summary_list


def PrintDeposit(title_and_summary_list):
    """Flatten the list"""
    flattened_list = []
    for item in title_and_summary_list:
        for value in item:
            flattened_list.append(value)
    return flattened_list


def initialize_feeds():
    """Initialize and process feeds"""
    global printdepositlist
    posts = fetch_and_parse_feeds()
    Only_the_titles_Summaries = OnlyTitlesandSummaries(posts)
    The_Title_Summary_List = TitleAndSummaryList(Only_the_titles_Summaries)
    printdepositlist = PrintDeposit(The_Title_Summary_List)
    return printdepositlist


# Initialize feeds when module is imported
printdepositlist = initialize_feeds()

if __name__ == "__main__":
    # Print to verify
    for entry in printdepositlist:
        print(entry)
    print("Print the length of printdepositlist")
    print(len(printdepositlist))

# Expose printdepositlist for import
__all__ = ['printdepositlist']
