# 1) Import posts from RssArticles_1
from RssArticles_1 import posts
import datetime
from datetime import datetime
import pandas as pd
import feedparser

def gettingNecessaryList():
 
    allitems = []
    
    #TODO: Replace with your actual code, e.g.:
    for x in posts:
        tempdict = {}
        tempdict["title"] = x.get("title", "")
        tempdict["summary"] = x.get("summary", "")
        tempdict["link"] = x.get("link", "")
        tempdict["published"] = x.get("published", "")
        allitems.append(tempdict)
    
    return allitems

#Store the list
AllItemsX = gettingNecessaryList()
#validate
print (AllItemsX)

from datetime import datetime

def ThefinalList():

    finalList = []
    
    # List of possible date formats to try
    possible_formats = [
        "%Y-%m-%d %H:%M:%S",     # e.g., 2025-02-03 15:30:00
        "%Y-%m-%dT%H:%M:%SZ",    # e.g., 2025-02-03T15:30:00Z (common in JSON APIs)
        "%a, %d %b %Y %H:%M:%S %z",  # e.g., Tue, 03 Feb 2025 15:30:00 +0000 (RFC 2822)
        "%Y-%m-%d"               # e.g., 2025-02-03 (if only the date is provided)
    ]
    
    for item in AllItemsX:
        # Extract values from item, defaulting to an empty string if not present
        title = item.get("title", "")
        summary = item.get("summary", "")
        link = item.get("link", "")
        published_raw = item.get("published", "")
        
        # Initialize the formatted published date with the raw value as a fallback
        published_formatted = published_raw
        
        # Try parsing the published date using each possible format
        for fmt in possible_formats:
            try:
                parsed_date = datetime.strptime(published_raw, fmt)
                published_formatted = parsed_date.strftime("%Y-%m-%d %H:%M:%S")
                break  # Stop trying once parsing succeeds
            except ValueError:
                continue  # Try the next format if the current one fails
        
        # Append the extracted and formatted data to finalList
        finalList.append([title, summary, link, published_formatted])
    
    return finalList


# Create and print the final list
MyTheFinalList = ThefinalList()

# 3) Create a variable that holds the final list
MyTheFinalList = ThefinalList()

#print the final list
#print(MyTheFinalList)
print(len(MyTheFinalList))