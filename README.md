# BK24TR Machine Learning - Assignment 3

├── **Group 5**
│ ├── Agne Dimsaite
│ ├── Ebrahim Amani
│ ├── Johan Mo
│ ├── Razan Kamal Taha Mohamed
│ ├── Therese Woods

# News Topic Prediction Pipeline

This repository contains a machine learning pipeline for predicting news topics from RSS feeds.
The system fetches RSS feeds, processes the text, trains and selects the best model, predicts news topics,
stores the data in an SQL database, and visualizes the results using Streamlit.

## Code Structure

```
├── src
│   ├── data_feeder.py
│   │   ├── text_preprocessor.py
│   │   ├── rss_feed_parser.py
│   │   ├── rss_feed_saver.py
│   ├── model_predictor.py
│   │   ├── text_preprocessor.py
│   ├── best_model_finder
│   │   ├── model_config.py
│   │   ├── model_manager.py
│   │   ├── model_persistence.py
│   │   ├── utils.py
│   ├── SQLConnectionTest.py
│   │   ├── db_connection.py
│   │   ├── dbconfig.ini
│   ├── sql_data_seeder.py
│   ├── streamlitUI.py
├── requirements.txt
├── .gitignore
```

## Workflow Overview

1. **Fetch RSS Feeds**

   - `data_feeder.py` retrieves news data from RSS feeds.
   - It preprocesses the text using `text_preprocessor.py`.
   - The parsed RSS data is saved using `rss_feed_saver.py`.

2. **Train and Select the Best ML Model**

   - `best_model_finder` contains scripts to configure, manage, and persist machine learning models.
   - `model_predictor.py` uses the trained model to predict news topics.

3. **Database Operations**

   - `SQLConnectionTest.py` ensures database connectivity.
   - `sql_data_seeder.py` inserts the predicted data into an SQL database.

4. **Data Visualization**

   - `streamlitUI.py` provides an interactive UI to visualize the results.

5. **Logging**

   - Logs are saved in files: `Mltraining.log`, `predictions.log`, `connection_test.log`, `dbconnection.log`.

## Installation & Setup

1. **Clone the repository**

   ```sh
   git clone git@github.com:Ebbi414/MLGroup5.git
   cd your-repo
   ```

2. **Create a virtual environment (optional but recommended)**

   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```sh
   pip install -r requirements.txt
   ```

4. **Set up the database**

   - dbconfig.ini file should be added containing the following data \
      [DATABASE]
     server = \<SQL Server Name>
     database = \<Database name>
     username = \<Username>
     password = \<Password>
     driver = ODBC Driver 17 for SQL Server

   - Modify `dbconfig.ini` with the correct database credentials.
   - Run `SQLConnectionTest.py` to verify the connection.

5. **Run the pipeline**

   ```sh
   python src/data_feeder.py
   python src/best_model_finder/model_manager.py
   python src/model_predictor.py
   python src/sql_data_seeder.py
   ```

6. **Run the Streamlit UI**

   ```sh
   streamlit run src/streamlitUI.py
   ```
