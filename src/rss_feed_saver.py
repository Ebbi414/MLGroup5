import json
from datetime import datetime
import os


class RSSFeedSaver:
    def __init__(self, parser):
        self.parser = parser

    def save_feeds(self):
        # Get feeds from the parser
        feeds = self.parser.fetch_and_parse_feeds()

        # Generate filename with current date and time
        current_time = datetime.now()
        filename = f"feeds_{current_time.strftime('%y%m%d_%H%M%S')}.json"

        # Create directory if it doesn't exist
        os.makedirs('feeds', exist_ok=True)
        filepath = os.path.join('feeds', filename)

        # Save feeds to JSON file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(feeds, f, indent=2, ensure_ascii=False)

        print(f"Saved {len(feeds)} feed entries to {filepath}")
        return filepath, len(feeds)
