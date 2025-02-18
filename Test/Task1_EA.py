import feedparser

# URL for the RSS feed
RSS_FEED_URL = [
    'http://www.dn.se/nyheter/m/rss/',
    'https://rss.aftonbladet.se/rss2/small/pages/sections/senastenytt/',
    'https://feeds.expressen.se/nyheter/',
    'http://www.svd.se/?service=rss',
    'http://api.sr.se/api/rss/program/83?format=145',
    'http://www.svt.se/nyheter/rss.xml'
]


# Fetch the RSS feeds and parse them
posts = []
for url in RSS_FEED_URL:
    feed = feedparser.parse(url)
    for entry in feed.entries:
        post = {
            "title": entry.get("title", ""),
            "summary": entry.get("description", ""),
        }
        posts.append(post)

def gettingNecessaryList():
        
    allitems = []

    posts = []
    for url in RSS_FEED_URL:
        feed = feedparser.parse(url)
        for entry in feed.entries:
            post = {
                "title": entry.get("title", ""),
                "summary": entry.get("description", ""),
            }
            posts.append(post)
    
    #TODO: Replace with your actual code, e.g.:
    for x in posts:
        tempdict = {}
        tempdict["title"] = x.get("title", "")
        tempdict["summary"] = x.get("summary", "")
        tempdict["link"] = x.get("link", "")
        tempdict["published"] = x.get("published", "")
        allitems.append(tempdict)
    
    return allitems

def OnlyTitlesandSummaries():
    """
    Extracts and returns a list of dictionaries containing only titles and summaries.
    Missing keys are replaced with empty strings.
    """
    only_titles_and_summaries = []
    for x in posts:
        tempdict = {
            "title": x.get("title", ""),  # Safely get 'title', default to ""
            # Safely get 'summary', default to ""
            "summary": x.get("summary", ""),
        }
        only_titles_and_summaries.append(tempdict)
    return only_titles_and_summaries


def TitleAndSummaryList(only_titles_and_summaries):
    """
    Combines titles and summaries into single strings for each post and wraps them in a list.
    """
    title_and_summary_list = []
    for item in only_titles_and_summaries:
        combined = item["title"] + " " + item["summary"]
        # Wrap combined string in a list
        title_and_summary_list.append([combined])
    return title_and_summary_list


def PrintDeposit(title_and_summary_list):
    """
    Flattens a nested list of combined titles and summaries into a single list.
    """
    flattened_list = []
    for item in title_and_summary_list:
        for value in item:  # Each item is a nested list
            flattened_list.append(value)
    return flattened_list


# -------------------- MAIN EXECUTION SECTION --------------------
if __name__ == "__main__":
    # 1. Extract only title and summary
    Only_the_titles_Summaries = OnlyTitlesandSummaries()

    # 2. Create nested lists of combined title+summary
    The_Title_Summary_List = TitleAndSummaryList(Only_the_titles_Summaries)

    # 3. Flatten and print the final result
    printdepositlist = PrintDeposit(The_Title_Summary_List)

    # Print to verify
    for entry in printdepositlist:
        print(entry)
