import subprocess
import time
import shlex
from pathlib import Path
import json
from bs4 import BeautifulSoup
from time import gmtime, strftime

import pandas as pd
import io
import cfscrape
import requests
from requests.adapters import HTTPAdapter
import os
from datetime import datetime, timedelta
import pytz

# from trading_calendars import get_calendar # Now unmaintained (https://github.com/quantopian/trading_calendars)
# from pandas_market_calendars import get_calendar # Alternative calendar library (https://github.com/rsheftel/pandas_market_calendars)
from exchange_calendars import get_calendar # New maintained fork of trading_calendars (https://github.com/gerrymanoim/exchange_calendars)

base_dir = Path(__file__).parent.absolute()
notebook_filestem = 'ark_fund_analysis'
notebook_path = base_dir / f'{notebook_filestem}.ipynb'
notebook_path_escaped = shlex.quote(str(notebook_path))

with open(base_dir / 'config.json') as file:
    config = json.load(file)

def download_fund_holdings_data():
    utc = pytz.utc
    est = pytz.timezone('US/Eastern')
    now_utc = datetime.now(utc)
    now_est = datetime.now(est)

    # US stock exchanges (includes NASDAQ)
    us_calendar = get_calendar('XNYS')

    timestamp_now = pd.Timestamp(now_utc)
    previous_close = us_calendar.previous_close(timestamp_now)
    next_close = us_calendar.next_close(timestamp_now)
    time_format = '%Y-%m-%d %H:%M:%S %z'

    base_dir = Path(__file__).parent.absolute()

    endpoint = 'https://ark-funds.com'
    adapter = HTTPAdapter(max_retries=5)
    session = requests.Session()
    session.mount(endpoint, adapter)

    fund_urls = {
        'arkk': 'https://ark-funds.com/wp-content/fundsiteliterature/csv/ARK_INNOVATION_ETF_ARKK_HOLDINGS.csv',
        'arkg': 'https://ark-funds.com/wp-content/fundsiteliterature/csv/ARK_GENOMIC_REVOLUTION_MULTISECTOR_ETF_ARKG_HOLDINGS.csv',
        'arkw': 'https://ark-funds.com/wp-content/fundsiteliterature/csv/ARK_NEXT_GENERATION_INTERNET_ETF_ARKW_HOLDINGS.csv',
        'arkf': 'https://ark-funds.com/wp-content/fundsiteliterature/csv/ARK_FINTECH_INNOVATION_ETF_ARKF_HOLDINGS.csv',
        'arkq': 'https://ark-funds.com/wp-content/fundsiteliterature/csv/ARK_AUTONOMOUS_TECHNOLOGY_&_ROBOTICS_ETF_ARKQ_HOLDINGS.csv',
        'arkx': 'https://ark-funds.com/wp-content/fundsiteliterature/csv/ARK_SPACE_EXPLORATION_&_INNOVATION_ETF_ARKX_HOLDINGS.csv',
    }

    fund_holdings_data_path = base_dir / 'data/ark_fund_holdings'

    print('Current time:                ' + str(now_utc.strftime(time_format)))
    print('Previous close:              ' + str(previous_close.astimezone(utc).strftime(time_format)))
    print('Next close:                  ' + str(next_close.astimezone(utc).strftime(time_format)))

    two_weeks_ago = now_utc - timedelta(days=14)
    prev_two_weeks_sessions = us_calendar.sessions_in_range(two_weeks_ago, now_utc)
    latest_session = max(prev_two_weeks_sessions)
    print(f'Latest session date:         {latest_session.date()}')

    latest_saved_dates = []

    def get_latest_saved_date(symbol):
        files = list(fund_holdings_data_path.glob('*.csv'))
        filestems = [file.stem for file in files if symbol in file.stem] # Filenames without parent directory or extension
        filestems_stripped = [i.replace(f'{symbol}_', '') for i in filestems] # Remove fund symbol
        dates = [datetime.strptime(i, '%Y_%m_%d') for i in filestems_stripped] # Convert to datetime objects
        if len(dates) > 0:
            latest_saved_date = max(dates)
            latest_saved_date = est.localize(latest_saved_date) # This is the correct way to append a time zone offset to a date without a time zone offset
            print(f'Latest saved {symbol.upper()} holdings data date: {str(latest_saved_date.date())}')
            return latest_saved_date
        else:
            return None

    def download_and_process(symbol, url):
        # Check saved files
        latest_saved_date = get_latest_saved_date(symbol)
        if latest_saved_date is not None:
            latest_saved_dates.append(latest_saved_date)

        if latest_saved_date is None or latest_saved_date.date() < latest_session.date():
            # Download CSV file (normal procedure, no longer works due to Cloudflare bot protection)
            # try:
            #     csv_bytes = session.get(url).content
            # except ConnectionError as connection_error:
            #     print(connection_error)

            # Use this to prevent bot blocking by Cloudflare
            scraper = cfscrape.create_scraper()
            csv_bytes = scraper.get(url).content

            df = pd.read_csv(io.StringIO(csv_bytes.decode('utf-8')))
            df.dropna(subset=['fund'], inplace=True) # Remove extraneous rows
            df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y') # Make date strings datetime objects
            downloaded_file_date = est.localize(df.iloc[0]['date'])
            print(f'Downloaded {symbol.upper()} holdings data date:   {str(downloaded_file_date.date())}')
            if latest_saved_date is None or latest_saved_date < downloaded_file_date:
                print(f'Saving downloaded {symbol.upper()} holdings data')
                filename = downloaded_file_date.strftime(f'{symbol}_%Y_%m_%d.csv')
                os.makedirs(fund_holdings_data_path, exist_ok=True)
                open(fund_holdings_data_path / filename, 'wb').write(csv_bytes) # Save
                latest_saved_dates.append(now_est)
            else:
                print(f'Saved {symbol.upper()} holdings data is up to date')
        else:
            print(f'Saved {symbol.upper()} holdings data is up to date')

    for symbol, url in fund_urls.items():
        download_and_process(symbol, url)
    return max(latest_saved_dates)

def export_notebook():
    # subprocess.call(f'{config["jupyter_path"]} nbconvert --to notebook --execute {notebook_path_escaped} --output output.ipynb', shell=True)
    # subprocess.call(f'{config["jupyter_path"]} nbconvert --to html {shlex.quote(str(notebooks_dir / "output.ipynb"))} --output {f"notebook_filestem"}.html', shell=True)
    # subprocess.call(f'{config["jupyter_path"]} nbconvert --to html --execute {notebook_path_escaped} --no-input --no-prompt --output {f"notebook_filestem"}.html', shell=True)
    try:
        start_time = time.time()
        subprocess.run(f'{config["jupyter_path"]} nbconvert --to html --execute {notebook_path_escaped} --no-input --no-prompt --output output/{notebook_filestem}.html', shell=True, check=True)
        # Replace title of HTML document
        with open(base_dir / f'output/{notebook_filestem}.html') as file:
            soup = BeautifulSoup(file, 'html.parser')
        soup.title.string.replace_with('ARK Fund Analysis')
        with open(base_dir / f'output/{notebook_filestem}.html', 'w') as file:
            file.write(str(soup))
        duration = time.time() - start_time # Seconds
        print(f"Notebook took {strftime('%H:%M:%S', gmtime(duration))} to convert")
    except subprocess.SubprocessError:
        raise

def download_fund_daily_price_data():
    funds = {'arkk': {}, 'arkg': {}, 'arkw': {}, 'arkf': {}, 'arkq': {}, 'arkx': {}}
    utc = pytz.utc
    us_calendar = get_calendar('XNYS')
    now_utc = datetime.utcnow().replace(tzinfo=utc) # Add time zone
    timestamp_now_utc = pd.Timestamp(now_utc)
    previous_close = us_calendar.previous_close(timestamp_now_utc) # UTC

    # one_day_ago = now_utc - timedelta(days=1)
    # timestamp_one_day_ago = pd.Timestamp(one_day_ago)
    # previous_close_from_one_day_ago = us_calendar.previous_close(timestamp_one_day_ago) # UTC

    # Alphavantage free tier rate limit: 5 requests per minute, 500 requests per day
    requests_per_minute = 5
    request_interval = 60 / requests_per_minute # 12 Seconds

    with open(base_dir / 'config.json') as file:
        config = json.load(file)

    alphavantage_api_key = config['alphavantage_api_key']
    alphavantage_api_endpoint = 'https://www.alphavantage.co/query'
    alphavantage_adapter = HTTPAdapter(max_retries=5)
    alphavantage_session = requests.Session()
    alphavantage_session.mount(alphavantage_api_endpoint, alphavantage_adapter) # Use `adapter` for all requests to endpoints that start with `alphavantage_api_endpoint`

    daily_dir = base_dir / 'data/ark_fund_daily_price_data'
    os.makedirs(daily_dir, exist_ok=True)

    def download_daily_data(symbol):
        try:
            print(f'Getting daily historical data for {symbol.upper()}')
            request_url = f'{alphavantage_api_endpoint}?function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}&outputsize=full&datatype=csv&apikey={alphavantage_api_key}'
            response = alphavantage_session.get(request_url)
            csv_bytes = response.content
            df = pd.read_csv(io.StringIO(csv_bytes.decode('utf-8')))
            if not df.empty and 'timestamp' in df.columns:
                df = df.set_index('timestamp')
                df.index = pd.to_datetime(df.index)
                print(f'Saving downloaded {symbol.upper()} data')
                open(daily_dir / f'{symbol}.csv', 'wb').write(csv_bytes) # Save
            else:
                print(f'No {symbol.upper()} data was returned.')
        except ConnectionError as connection_error:
            print(connection_error)

    # Download full daily historical data
    for i, fund in enumerate(funds):
        wait = True
        filename = f'{fund}.csv'
        if (daily_dir / filename).is_file():
            df = pd.read_csv(daily_dir / filename, parse_dates=['timestamp'], index_col='timestamp')
            latest_saved_data = max(df.index).date()
            if latest_saved_data < previous_close.date():
                print(f'Daily data for {fund.upper()} needs to be updated')
                download_daily_data(symbol=fund)
            else:
                print(f'Daily data for {fund.upper()} is up to date')
                wait = False
        else:
            download_daily_data(symbol=fund)
        if i != len(funds.keys()) - 1 and wait is True:
            time.sleep(request_interval)
