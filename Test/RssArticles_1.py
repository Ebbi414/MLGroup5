# Import required package
import feedparser

#############################  RSS FEED Parser  ##############################

# Define RSS feed URLs
RSS_URLS = ['http://www.dn.se/nyheter/m/rss/',
            'https://rss.aftonbladet.se/rss2/small/pages/sections/senastenytt/', 'https://feeds.expressen.se/nyheter/',
            'http://www.svd.se/?service=rss', 'http://api.sr.se/api/rss/program/83?format=145',
            'http://www.svt.se/nyheter/rss.xml'
              ]  # Example URLs

# Initialize an empty list to store the parsed articles
posts = []

# Loop through each RSS feed URL
for url in RSS_URLS:
    try:
        # Parse the RSS feed
        feed = feedparser.parse(url)

        # Check if the feed is parsed successfully
        if feed.bozo:  # bozo flag indicates an issue in parsing
            print(f"Error parsing URL: {url}")
            continue

        # Loop through each entry in the feed
        for entry in feed.entries:
            # Extract relevant fields from the entry
            article = {
                'title': entry.get('title', 'No title'),
                'link': entry.get('link', 'No link'),
                'published': entry.get('published', 'No date'),
                'summary': entry.get('summary', 'No summary'),
                'source': url  # Add source URL for reference
            }

            # Append the structured article to the posts list
            posts.append(article)

    except Exception as e:
        print(f"An error occurred while processing the URL {url}: {e}")

# Display the number of articles retrieved
print(f"Successfully retrieved {len(posts)} articles.")
print ;posts