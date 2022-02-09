import pandas as pd
import os
import requests
from requests.adapters import HTTPAdapter
from datetime import datetime
import pytz
from dateutil.relativedelta import relativedelta
# from trading_calendars import get_calendar # Now unmaintained (https://github.com/quantopian/trading_calendars)
# from pandas_market_calendars import get_calendar # Alternative calendar library, lacks some functionality (https://github.com/rsheftel/pandas_market_calendars)
from exchange_calendars import get_calendar # New maintained fork of trading_calendars (https://github.com/gerrymanoim/exchange_calendars)
import json
import math
from pathlib import Path
from colour import Color
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tabula
import camelot
import re

base_dir = Path(__file__).parent.absolute()

utc = pytz.UTC
now_utc = datetime.utcnow().replace(tzinfo=utc) # Add time zone
timestamp_now = pd.Timestamp(now_utc).replace(second=0, microsecond=0) # Seconds and microseconds must be 0 for exchange_calendars library
us_calendar = get_calendar('XNYS')
previous_close = us_calendar.previous_close(timestamp_now)
time_format = '%Y-%m-%d %H:%M:%S %z'

funds = {'arkk': {}, 'arkg': {}, 'arkw': {}, 'arkf': {}, 'arkq': {}, 'arkx': {}}

column_names = {
    'date_x': 'Date 1',
    'date_y': 'Date 2',
    'shares_x': 'Shares 1',
    'shares_y': 'Shares 2',
    'company': 'Company',
    'company_x': 'Company',
    'ticker': 'Symbol',
    'shares': 'Shares',
    'market value ($)': 'Market value',
    'market value ($)_x': 'Market value 1',
    'market value ($)_y': 'Market value 2',
    'weight (%)': 'Weight',
    'weight (%)_x': 'Weight 1',
    'weight (%)_y': 'Weight 2',
    'fund_x': 'Fund 1',
    'fund_y': 'Fund 2',
    'change_in_weight': 'Change in weight',
    'relative_change_in_weight': 'Relative change in weight',
    'change_in_shares': 'Change in shares',
    'percent_change_in_shares': 'Change in shares',
    'relative_change': 'Relative change',
    'share_price': 'Share price',
    'share_price_x': 'Share price 1',
    'share_price_y': 'Share price 2',
    'change_in_share_price': 'Change in share price',
    'change_in_value': 'Change in value',
    'fund_inflow_outflow': 'Estimated fund inflow/outflow',
    'percent_ownership': 'Ownership',
    'market_cap': 'Market cap (billions)',
    'pe_ratio': 'P/E ratio',
    'arkk': 'ARKK',
    'arkw': 'ARKW',
    'arkq': 'ARKQ',
    'arkg': 'ARKG',
    'arkf': 'ARKF',
    'arkx': 'ARKX',
    'category': 'Category',
    'unique': 'Not in ARKK',
    'contribution': 'Contribution',
}

# Python string formatting for numbers: https://mkaz.blog/code/python-string-format-cookbook/
int_comma_sep = '{:,.0f}' # Round to integer, comma separator
dollars_int = '${:,.0f}' # Dollar sign, comma separator, round to integer
perc_two_dec = '{:.2%}' # Percent, round to two decimal places
perc_two_dec_sign = '{:+.2%}' # Percent, round to two decimal places, sign
dollars_two_dec = '${:,.2f}' # Dollar sign, comma separator, round to two decimal places
number_formats = {
    column_names['shares']: int_comma_sep,
    column_names['market value ($)']: dollars_int,
    column_names['weight (%)']: perc_two_dec,
    column_names['shares']: int_comma_sep,
    column_names['market value ($)']: dollars_int,
    column_names['shares_x']: int_comma_sep,
    column_names['market value ($)_x']: dollars_int,
    column_names['weight (%)_x']: perc_two_dec,
    column_names['shares_y']: int_comma_sep,
    column_names['market value ($)_y']: dollars_int,
    column_names['weight (%)_y']: perc_two_dec,
    column_names['change_in_weight']: perc_two_dec_sign,
    column_names['relative_change']: perc_two_dec_sign,
    column_names['change_in_shares']: '{:+,.0f}', # Round to integer, comma separator, sign
    column_names['relative_change_in_weight']: '{:+.1%}', # Percent, round to one decimal place, sign
    column_names['percent_change_in_shares']: perc_two_dec_sign,
    column_names['share_price']: dollars_two_dec,
    column_names['share_price_x']: dollars_two_dec,
    column_names['share_price_y']: dollars_two_dec,
    column_names['change_in_share_price']: perc_two_dec_sign,
    column_names['change_in_value']: perc_two_dec_sign,
    column_names['fund_inflow_outflow']: perc_two_dec_sign,
    column_names['percent_ownership']: perc_two_dec,
    column_names['market_cap']: '${:,.1f}', # Dollar sign, comma separator, round to one decimal place
    column_names['pe_ratio']: '{:,.1f}', # Comma separator, round to one decimal place
    column_names['contribution']: perc_two_dec_sign,
    column_names['ticker']: str.upper,
    column_names['arkk']: str.upper,
    column_names['arkw']: str.upper,
    column_names['arkg']: str.upper,
    column_names['arkf']: str.upper,
    column_names['arkq']: str.upper,
    column_names['arkx']: str.upper,
}

category_labels = {
    'finance': ['pypl', 'lc', 'sq', 'tree', 'ice', 'z', 'ipob', 'adyey', 'gbtc', 'beke'],
    'consumer': ['aapl', 'amzn', 'meli', 'baba', 'tcehy', 'tsla', 'nflx', 'spot', 'roku', 'goog', 'se', 'jd', 'shop'],
    'cloud': ['work', 'twlo', 'nvda', 'tsm', 'splk', 'fsly', 'team', 'adbe', 'net', 'wix', 'pltr', 'zm', 'bidu', 'pstg', 'docu', 'snps', 'pd', 'crwd', 'okta', 'api'],
    'social': ['snap', 'fb', 'pins', 'twtr'],
    'gaming': ['ntdoy', 'u', 'doyu', 'huya'],
    'healthcare': ['tdoc'],
    'biotech': ['vcyt', 'onvo'],
    'education': ['twou'],
    'marketing': ['ttd', 'hubs'],
}

symbols_with_spaces = []

def map_csv_to_df(csv_file):
    fund_re = re.compile(r'.+?(?=_)') # Matches text before the first underscore
    fund = fund_re.match(csv_file.stem).group()
    if csv_file.stem in funds[fund]['dfs']:
        return funds[fund]['dfs'][csv_file.stem]
    else:
        df = csv_to_df(csv_file)
        funds[fund]['dfs'][csv_file.stem] = df.copy()
        return funds[fund]['dfs'][csv_file.stem]

def csv_to_df(csv_file):
    df = pd.read_csv(csv_file)
    df_date = pd.to_datetime(df['date'][0], format='%m/%d/%Y').date()
    df_fund = df['fund'][0].lower()
    # Add `shares` column from PDF if necessary (share counts not included in CSV files for 4 and 5 October 2021)
    if 'shares' not in df.columns:
        # Check for cached CSV with `shares` column from PDF
        combined_csv_path = base_dir / 'data/ark_fund_holdings/csv_with_shares' / df_date.strftime(f'{df_fund}_%Y_%m_%d.csv')
        if os.path.isfile(combined_csv_path):
            # Use existing file
            df = pd.read_csv(combined_csv_path)
        else:
            # Process PDF and use share count from PDF
            pdf_path = base_dir / 'data/ark_fund_holdings/pdf' / df_date.strftime(f'{df_fund}_%Y_%m_%d.pdf')
            if not os.path.isfile(pdf_path):
                raise Exception('The CSV file does not contain the `shares` column, and no corresponding PDF file exists.')
            else:
                tabula_tables = tabula.read_pdf(pdf_path, pages='all')
                # tabula correctly parses the column headers on the second page, but not the first page
                column_headers = tabula_tables[1].set_index(keys='#').columns
                camelot_tables = camelot.read_pdf(str(pdf_path), pages='all')
                full_df = pd.concat([table.df.set_index(keys=0) for table in camelot_tables])
                full_df = full_df.set_axis(column_headers, axis=1) # Add column headers from tabula
                full_df.index.names = ['']
                # Rename columns to match CSV files
                full_df.rename(columns={'CUSIP': 'cusip', 'Shares': 'shares'}, inplace=True)
                # Convert `Shares` column to integers
                full_df['shares'] = full_df['shares'].apply(lambda x: pd.to_numeric(x.replace(',', '')))
                partial_df = full_df[['cusip', 'shares']].copy()
                df = pd.merge(df, partial_df, on=['cusip'], how='outer')
                # Save cached copy of CSV file with shares column from PDF
                df.to_csv(combined_csv_path)
    # Rename columns if necessary (naming in CSV files changed on 4 October 2021)
    if 'market value($)' in df.columns:
        df.rename(columns={'market value($)': 'market value ($)'}, inplace=True)
    if 'weight(%)' in df.columns:
        df.rename(columns={'weight(%)': 'weight (%)'}, inplace=True)
    # Use lowercase symbols in code and uppercase for display
    for index, row in df.iterrows():
        if not pd.isnull(df.loc[index, 'ticker']):
            # Remove endings and whitespace from symbols
            df.loc[index, 'ticker'] = df.loc[index, 'ticker'].strip().lower() # Remove trailing and leading whitespace from symbols, and make lowercase
            for ending in [' uw', ' uq', ' un', ' u']:
                if df.loc[index, 'ticker'].endswith(ending):
                    df.loc[index, 'ticker'] = df.loc[index, 'ticker'].rstrip(ending) # Remove ending from end of string
            # Custom fixes
            if df.loc[index, 'ticker'] == 'dsy' and df.loc[index, 'company'] == 'DISCOVERY LTD':
                df.loc[index, 'ticker'] = 'dsy.jo' # Disambiguate Dassault Systems and Discovery Limited (ARK's data uses same symbol DSY for both)
            if df.loc[index, 'ticker'] == 'tcs li':
                df.loc[index, 'ticker'] = 'tcs.li'
            if df.loc[index, 'ticker'] == 'dsy fp':
                df.loc[index, 'ticker'] = 'dsy.fp'
            # Check for other cases not yet accounted for
            if ' ' in df.loc[index, 'ticker']:
                symbols_with_spaces.append(df.loc[index, 'ticker'])
    # Aggregate multiple rows with same asset (e.g. Japanese yen)
    aggregate_functions = {'date': 'first', 'fund': 'first', 'company': 'first', 'ticker': 'first', 'cusip': 'first', 'shares': 'sum', 'market value ($)': 'sum', 'weight (%)': 'sum'}
    df = df.groupby(df['cusip'], as_index=False).aggregate(aggregate_functions).sort_values(by=['weight (%)'], ascending=False).reset_index(drop=True)
    df.dropna(subset=['fund'], inplace=True)
    df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y').dt.date # Convert to datetime object and display only date without time
    # Some columns are formatted as strings in the CSV files starting on 4 October 2021. These need to be converted to floats.
    if 'shares' in df and df['shares'].dtype == object and isinstance(df.iloc[0]['shares'], str):
        df['shares'] = df['shares'].apply(lambda x: pd.to_numeric(x.replace(',', '')))
    if df['market value ($)'].dtype == object and isinstance(df.iloc[0]['market value ($)'], str):
        df['market value ($)'] = df['market value ($)'].apply(lambda x: pd.to_numeric(x.replace(',', '').replace('$', '')))
    if df['weight (%)'].dtype == object and isinstance(df.iloc[0]['weight (%)'], str):
        df['weight (%)'] = df['weight (%)'].apply(lambda x: pd.to_numeric(x.replace('%', '')))
    total_value = df['market value ($)'].sum()
    df['share_price'] = df['market value ($)'] / df['shares']
    df['weight (%)'] = df['market value ($)'] / total_value # Recalculate to remove rounding errors
    df = df.sort_values(by=['weight (%)'], ascending=False).reset_index(drop=True)
    return df

def split_into_batches(full_list, batch_size):
    batches = []
    for i in range(0, math.ceil(len(full_list) / batch_size)):
        batches.append(full_list[(i * batch_size):((i + 1) * batch_size)])
    return batches

def add_totals(df):
    df_copy = df.copy()
    if 'weight (%)_x' in df_copy.columns:
        df_copy.loc['Total', 'weight (%)_x'] = df_copy['weight (%)_x'].sum()
        df_copy.loc['Total', 'market value ($)_x'] = df_copy['market value ($)_x'].sum()
        df_copy.loc['Total', 'weight (%)_y'] = df_copy['weight (%)_y'].sum()
        df_copy.loc['Total', 'market value ($)_y'] = df_copy['market value ($)_y'].sum()
    elif 'weight (%)' in df_copy.columns:
        df_copy.loc['Total', 'weight (%)'] = df_copy['weight (%)'].sum()
        df_copy.loc['Total', 'market value ($)'] = df_copy['market value ($)'].sum()
    return df_copy

def process_for_change_in_holdings(df1, df2, fund):
    start_date = df1['date'].dropna().iloc[0]
    end_date = df2['date'].dropna().iloc[0]
    '''
    Starting on 15 March 2021, ARK began reporting its holdings at the beginning of the trading day,
    instead of at the end of the trading day.
    From this date on, the date on the CSV file refers to the beginning of the respective trading day.
    Before that date, the date on the CSV file refers to the end of the respective trading day.
    For this reason, CSV files for 12 and 15 March 2021 contain the same data.
    '''
    switchover = datetime(2021, 3, 15).date()
    # Not merging on 'company' column, because sometimes company names change in CSV files
    merged = pd.merge(df1, df2, on=['fund', 'cusip', 'ticker'], how='outer') # `how='outer'` keeps non-matching rows
    # Use most recent company name from CSVs
    for index, row in merged.iterrows():
        if pd.notna(row['company_y']):
            merged.loc[index, 'company_x'] = row['company_y'] # Using `company_x` to display, so need to copy more recent name
        if row['ticker'] in stock_splits:
            if stock_splits[row['ticker']]['date'].date() >= switchover:
                # Move ahead one day to adjust for ARK's dating of CSV files (see note above)
                split_date_adjusted = us_calendar.next_open(stock_splits[row['ticker']]['date'].date() + pd.DateOffset(1)).date()
            else:
                split_date_adjusted = stock_splits[row['ticker']]['date'].date()
            if split_date_adjusted > start_date and split_date_adjusted <= end_date:
                merged.loc[index, 'split_factor'] = stock_splits[row['ticker']]['factor']
                merged.loc[index, 'shares_y'] /= merged.loc[index, 'split_factor']
                merged.loc[index, 'share_price_y'] *= merged.loc[index, 'split_factor']
    merged['change_in_share_price'] = (merged['share_price_y'] - merged['share_price_x']) / merged['share_price_x']
    merged['change_in_value'] = (merged['market value ($)_y'] - merged['market value ($)_x']) / merged['market value ($)_x']
    merged['change_in_weight'] = merged['weight (%)_y'] - merged['weight (%)_x']
    merged['relative_change_in_weight'] = merged['change_in_weight'] / merged['weight (%)_x']
    merged['change_in_shares'] = merged['shares_y'] - merged['shares_x']
    merged['percent_change_in_shares'] = merged['change_in_shares'] / merged['shares_x']
    for index, row in merged.iterrows():
        if pd.isna(row['weight (%)_x']):
            merged.loc[index, 'sort_rank'] = 1000000
        elif pd.isna(row['weight (%)_y']):
            merged.loc[index, 'sort_rank'] = -1000000
        else:
            merged.loc[index, 'sort_rank'] = merged.loc[index, 'percent_change_in_shares']
    merged = add_totals(merged)
    merged = merged.sort_values(by=['sort_rank', 'percent_change_in_shares'], ascending=False).reset_index(drop=True)
    merged.rename(index={merged.index[-1]: 'Total'}, inplace=True) # Need to rename last row of index after resetting index
    if 'split_factor' in merged.columns:
        merged = merged.drop(['split_factor'], axis=1)
    merged.loc['Total', 'change_in_value'] = (merged.loc['Total', 'market value ($)_y'] - merged.loc['Total', 'market value ($)_x']) / merged.loc['Total', 'market value ($)_x']
    if start_date < switchover:
        start_previous_close = start_date
    else:
        start_previous_close = us_calendar.previous_close(pd.Timestamp(start_date)).date()
    if end_date < switchover:
        end_previous_close = end_date
    else:
        end_previous_close = us_calendar.previous_close(pd.Timestamp(end_date)).date()
    if end_previous_close in funds[fund]['daily_price_df'].index:
        if start_previous_close in funds[fund]['daily_price_df'].index:
            merged.loc['Total', 'change_in_share_price'] = (funds[fund]['daily_price_df'].loc[end_previous_close]['close'] - funds[fund]['daily_price_df'].loc[start_previous_close]['close']) / funds[fund]['daily_price_df'].loc[start_previous_close]['close']
        else:
            merged.loc['Total', 'change_in_share_price'] = 0
        merged.loc['Total', 'fund_inflow_outflow'] = merged.loc['Total', 'change_in_value'] - merged.loc['Total', 'change_in_share_price']
    merged = merged.drop(['fund', 'cusip', 'sort_rank', 'change_in_shares', 'market value ($)_x', 'market value ($)_y', 'change_in_weight', 'relative_change_in_weight', 'shares_x', 'shares_y', 'share_price_x', 'share_price_y', 'company_y'], axis=1)
    return merged

def negative_red_hide_empty(val):
    if isinstance(val, float) and val < 0:
        return 'color: red'
    elif pd.isnull(val):
        return 'opacity: 0'

def total_row_bold(df):
    return ['font-weight: bold' if row == df.loc['Total'] else '' for row in df]

def apply_style(df):
    # Necessary for string formatting of entire column. Make na values empty string
    if 'ticker' in df:
        df['ticker'] = df.apply(lambda row: '' if pd.isna(row['ticker']) else row['ticker'], axis=1)
    if df.index.name == 'Symbol':
        df.index = df.index.map(str.upper)
    styled = df.rename(columns=column_names).style.format(number_formats).applymap(negative_red_hide_empty).set_properties(**{'background-color': ''})
    if 'Total' in df.index:
        styled = styled.apply(total_row_bold)
    if 'unique_weight' in df.columns:
        styled = styled.hide(axis='columns')
    return styled

def clamp(n, min_val, max_val):
    return max(min_val, min(n, max_val))

def date_to_datetime_with_timezone(date):
    return datetime(date.year, date.month, date.day).replace(tzinfo=utc) # Alternative method: return pd.Timestamp(date).tz_localize(utc)

def process_for_change_in_value(change_in_holdings_df):
    min_lum = 0.45
    max_lum = 0.95
    lum_diff = max_lum - min_lum
    green = Color('green')
    green.saturation = 0.9
    green.luminance = max_lum
    red = Color('red')
    red.saturation = 0.9
    red.luminance = max_lum
    clamp_threshold = 0.2
    change_in_value_df = change_in_holdings_df[['company_x', 'ticker', 'weight (%)_x', 'change_in_share_price', 'change_in_value']].drop(index='Total').copy()
    change_in_value_df['contribution'] = change_in_value_df['weight (%)_x'] * change_in_value_df['change_in_share_price']
    change_in_value_df['contribution_abs'] = abs(change_in_value_df['contribution'])
    for index, row in change_in_value_df.iterrows():
        change_in_share_price = row['change_in_share_price']
        cisp_clamp_abs = abs(clamp(change_in_share_price, -clamp_threshold, clamp_threshold))
        if change_in_share_price >= 0:
            color = green
            color.luminance = max_lum - (lum_diff * (cisp_clamp_abs / clamp_threshold))
        else:
            color = red
            color.luminance = max_lum - (lum_diff * (cisp_clamp_abs / clamp_threshold))
        change_in_value_df.loc[index, 'color'] = color.hex
    change_in_value_df = change_in_value_df.sort_values(by=['contribution'], ascending=False).reset_index(drop=True)
    change_in_value_df = change_in_value_df.reset_index(drop=True) # Sort all rows except last row (totals)
    change_in_value_df.loc['Total', 'weight (%)_x'] = change_in_value_df['weight (%)_x'].sum()
    change_in_value_df.loc['Total', 'contribution'] = change_in_value_df['contribution'].sum()
    change_in_value_df.loc['Total', 'change_in_value'] = change_in_holdings_df.loc['Total', 'change_in_value']
    change_in_value_df.loc['Total', 'change_in_share_price'] = change_in_holdings_df.loc['Total', 'change_in_share_price']
    return change_in_value_df

def nearest(items, pivot):
    return min(items, key=lambda x: abs(x - pivot))

def batch_str_format(symbols):
    return ','.join(symbols)

def process_for_sort_by_weight(df):
    df_copy = df.sort_values(by=['weight (%)'], ascending=False)
    df_copy = add_totals(df_copy)
    df_copy = df_copy.drop(['fund', 'date', 'cusip', 'share_price'], axis=1)
    return df_copy

def process_percent_ownership(df):
    for index, row in df.iterrows():
        ticker = df.loc[index, 'ticker']
        # IEX Cloud
        if pd.notna(row['ticker']) and ticker in all_stocks_info and all_stocks_info[ticker]['stats'] is not None and all_stocks_info[ticker]['stats']['sharesOutstanding'] != 0 and all_stocks_info[ticker]['stats']['marketcap'] is not None:
            df.loc[index, 'market_cap'] = all_stocks_info[ticker]['stats']['marketcap'] / 1000000000
            df.loc[index, 'shares_outstanding'] = all_stocks_info[ticker]['stats']['sharesOutstanding']
            df.loc[index, 'percent_ownership'] = df.loc[index, 'shares'] / df.loc[index, 'shares_outstanding']
            df.loc[index, 'total_ownership_contribution'] = df.loc[index, 'weight (%)'] * df.loc[index, 'percent_ownership']
            df.loc[index, 'pe_ratio'] = all_stocks_info[ticker]['stats']['peRatio']
    df.loc['Total', 'total_ownership_contribution'] = df['total_ownership_contribution'].sum()
    df.loc['Total', 'percent_ownership'] = df.loc['Total', 'total_ownership_contribution']
    df = df.drop(['shares_outstanding', 'total_ownership_contribution'], axis=1)
    return df

def create_share_changes_df(fund, symbol, n):
    previous_n_shares = None
    df_data = {'date': [], 'shares': [], 'perc_shares_change': [], 'n_shares_change': []}
    csv_subset = funds[fund]['csv_files'][-n:-1]
    for i, csv_file in enumerate(csv_subset):
        df = map_csv_to_df(csv_subset[i]).set_index('ticker')
        if symbol in df.index:
            df_data['date'].append(df.loc[symbol, 'date'])
            df_data['shares'].append(df.loc[symbol, 'shares'])
            if previous_n_shares is not None:
                n_shares_change = df.loc[symbol, 'shares'] - previous_n_shares
                perc_shares_change = n_shares_change / previous_n_shares
                previous_n_shares = df.loc[symbol, 'shares']
            else:
                if i == 0:
                    # Show no change for first date available (!! assumes first acquisition of stock was not made on this day, which may not be true)
                    n_shares_change = None
                else:
                    n_shares_change = df.loc[symbol, 'shares']
                perc_shares_change = None
                previous_n_shares = df.loc[symbol, 'shares']
            df_data['perc_shares_change'].append(perc_shares_change)
            df_data['n_shares_change'].append(n_shares_change)
        else:
            df_data['date'].append(df.iloc[0]['date'])
            # !! Using 0 rather than None in the next three lines produces some warnings, but using None produces an error. Fix later.
            df_data['shares'].append(0)
            df_data['perc_shares_change'].append(0)
            df_data['n_shares_change'].append(0)
    stock_purchases_df = pd.DataFrame.from_dict(df_data, orient='columns').set_index('date')
    return stock_purchases_df

def process_unique_holdings(df):
    arkk_comparison_df = funds['arkk']['holdings_by_weight_df'].set_index('ticker')
    unique_holdings = 0
    comparison_df = df.set_index('ticker')
    comparison_df = comparison_df[comparison_df['company'].notna()] # Only keep rows with company name (removes 'Total' row)
    for symbol in comparison_df.index:
        if pd.notna(symbol) and symbol not in arkk_comparison_df.index:
            unique_holding_weight = comparison_df.loc[symbol, 'weight (%)']
            unique_holdings += unique_holding_weight
            comparison_df.loc[symbol, 'unique_weight'] = unique_holding_weight
            comparison_df.loc[symbol, 'unique'] = True
    comparison_df = comparison_df.reset_index()
    comparison_df = add_totals(comparison_df)
    comparison_df.loc['Total', 'unique_weight'] = comparison_df['unique_weight'].sum()
    comparison_df.loc['Total', 'unique'] = comparison_df.loc['Total', 'unique_weight']
    comparison_df.loc['Total', 'unique'] = '{:.2%}'.format(comparison_df.loc['Total', 'unique'])
    return comparison_df

def show_share_change_graph(fund):
    columns = 3
    days_to_display = 40
    if len(funds[fund]['csv_files']) >= 2:
        # !! Should also include symbols not in latest holdings_by_weight_df (create same df for 40 sessions ago)
        symbols = [symbol for symbol in funds[fund]['holdings_by_weight_df']['ticker'] if symbol != '']
        rows = math.ceil(len(symbols) / columns)
        df = funds[fund]['holdings_by_weight_df'].set_index('ticker')
        height = 4 * rows
        fig = plt.figure(figsize=(22, height), tight_layout=True)
        spec = gridspec.GridSpec(ncols=columns, nrows=rows, figure=fig)
        for i, symbol in enumerate(symbols):
            row = math.floor(i / columns)
            column = i % columns
            company = df.loc[symbol, 'company']
            weight = df.loc[symbol, 'weight (%)']
            share_changes_df = create_share_changes_df(fund, symbol, days_to_display)
            title = f'{fund.upper()}: {company} ({symbol.upper()}), weight: {round(weight * 100, 2)}%'
            ax = fig.add_subplot(spec[row, column])
            share_changes_df.plot(kind='bar', width=0.9, ax=ax, y='n_shares_change', title=title, legend=False).xaxis.label.set_visible(False)
            max_shares = share_changes_df['shares'].max()
            ax.set_ylim([-max_shares, max_shares]) # Use maximum number of shares held during time period as +/- limits for y axis
            # ax.figure.autofmt_xdate() # Removes date labels from x axis except in last row
            ax.ticklabel_format(scilimits=(0, 0), axis='y') # Always use scientific notation on y axis (more compact)
        plt.show()

def csv_path_to_date(csv_path):
    stem = csv_path.stem # Filename without parent directory or extension
    fund_re = re.compile(r'.+?(?=_)') # Matches text before the first underscore
    fund = fund_re.match(csv_path.stem).group()
    stripped = stem.replace(f'{fund}_', '')
    date = datetime.strptime(stripped, '%Y_%m_%d').date()
    return date

def plot_share_price_and_estimated_capital_flows(fund, start_date=None):
    cached_data_path = base_dir / f'output/{fund}_share_price_and_estimated_capital_flows.csv'
    if os.path.isfile(cached_data_path):
        cached_df = pd.read_csv(cached_data_path)
        cached_df['date'] = pd.to_datetime(cached_df['date'], format='%Y-%m-%d').dt.date # Convert to datetime object and display only date without time
    else:
        cached_df = None
    change_in_holdings_dfs = []
    change_in_value_dfs = []
    data = {
        'date': [],
        'change_in_share_price': [],
        'change_in_value': [],
        'estimated_change_in_share_price': [],
        'estimated_capital_flows': [],
        'share_price_cumulative': [],
        'estimated_capital_flows_cumulative': [],
    }
    for i, csv_file in enumerate(funds[fund]['csv_files']):
        if (cached_df is None or csv_path_to_date(csv_file) not in cached_df['date'].values) and (start_date is None or csv_path_to_date(csv_file) >= start_date):
            df = map_csv_to_df(csv_file)
            if start_date is None:
                start_date = funds[fund]['earliest_date_from_data']
            if df['date'].dropna().iloc[0] >= start_date:
                if i == 0:
                    data['date'].append(df['date'].dropna().iloc[0])
                    data['change_in_share_price'].append(None)
                    data['change_in_value'].append(None)
                    data['estimated_change_in_share_price'].append(None)
                    data['estimated_capital_flows'].append(None)
                elif i > 0 and i < len(funds[fund]['csv_files']):
                    change_in_holdings_df = process_for_change_in_holdings(map_csv_to_df(funds[fund]['csv_files'][i - 1]), map_csv_to_df(funds[fund]['csv_files'][i]), fund)
                    change_in_holdings_dfs.append(change_in_holdings_df)
                    change_in_value_df = process_for_change_in_value(change_in_holdings_df)
                    change_in_value_dfs.append(change_in_value_df)
                    # Data for new dataframe
                    data['date'].append(change_in_holdings_df['date_y'].dropna().iloc[0])
                    data['change_in_share_price'].append(change_in_holdings_df.loc['Total', 'change_in_share_price'])
                    data['change_in_value'].append(change_in_value_df.loc['Total', 'change_in_value'])
                    data['estimated_change_in_share_price'].append(change_in_value_df.loc['Total', 'contribution'])
                    if 'fund_inflow_outflow' in change_in_holdings_df:
                        data['estimated_capital_flows'].append(change_in_holdings_df.loc['Total', 'fund_inflow_outflow'])
                    else:
                        data['estimated_capital_flows'].append(None)
    for i, item in enumerate(data['change_in_share_price']):
        if i == 0:
            if cached_df is None:
                data['share_price_cumulative'].append(1)
                data['estimated_capital_flows_cumulative'].append(1)
            else:
                data['share_price_cumulative'].append(cached_df['share_price_cumulative'].iloc[-1] * (1 + data['change_in_share_price'][i]))
                data['estimated_capital_flows_cumulative'].append(cached_df['estimated_capital_flows_cumulative'].iloc[-1] * (1 + data['estimated_capital_flows'][i]))
        else:
            data['share_price_cumulative'].append(data['share_price_cumulative'][i - 1] * (1 + data['change_in_share_price'][i]))
            if pd.notna(data['estimated_capital_flows'][i]):
                data['estimated_capital_flows_cumulative'].append(data['estimated_capital_flows_cumulative'][i - 1] * (1 + data['estimated_capital_flows'][i]))
            else:
                data['estimated_capital_flows_cumulative'].append(data['estimated_capital_flows_cumulative'][i - 1])
    df = pd.DataFrame.from_dict(data)
    df['difference'] = df['change_in_share_price'] - df['estimated_change_in_share_price']
    if cached_df is not None:
        df = pd.concat([cached_df, df])
    # df[['difference']].plot.line(y='difference', figsize=(20, 5)) # Check if estimated change in share price based on data is within ordinary margin of error
    funds[fund]['estimated_capital_flows_and_share_price_df'] = df.copy()
    df.to_csv(cached_data_path, index=False) # Cache as CSV
    df.set_index('date', drop=True, inplace=True)
    df.index.names = ['']
    df[['share_price_cumulative', 'estimated_capital_flows_cumulative']].rename(columns={'share_price_cumulative': 'Share price', 'estimated_capital_flows_cumulative': 'Estimated capital flows'}).plot.line(title=f'{fund.upper()}', figsize=(20, 5))
    plt.legend(loc='upper left')
    plt.show()

# =====

fund_holdings_csv_path = base_dir / 'data/ark_fund_holdings/csv'
csv_files = list(fund_holdings_csv_path.glob('*.csv'))

for fund in funds:
    funds[fund]['csv_files'] = sorted([path for path in csv_files if fund in str(path)])
    funds[fund]['dfs'] = {}
    companies_data = {'symbol': [], 'company': []}
    latest_df = map_csv_to_df(funds[fund]['csv_files'][-1])
    for index, row in latest_df.iterrows():
        ticker = latest_df.loc[index, 'ticker']
        if not pd.isna(ticker):
            companies_data['symbol'].append(ticker)
            companies_data['company'].append(latest_df.loc[index, 'company'])
    funds[fund]['companies_df'] = pd.DataFrame.from_dict(companies_data, orient='columns').set_index('symbol')
    funds[fund]['daily_price_df'] = pd.read_csv(base_dir / f'data/ark_fund_daily_price_data/{fund}.csv').set_index('timestamp')
    funds[fund]['daily_price_df'].index = pd.to_datetime(funds[fund]['daily_price_df'].index, format='%Y-%m-%d').date # Convert to datetime object and display only date without time

symbols_with_spaces = set(symbols_with_spaces)
if len(symbols_with_spaces) > 0:
    print(f'Additional fixes may need to be made for the following symbols with spaces: {symbols_with_spaces}')

all_companies_df_elements = []
for fund in funds:
    all_companies_df_elements.append(funds[fund]['companies_df'])
all_companies_df = pd.concat(all_companies_df_elements).drop_duplicates()
all_symbols = list(all_companies_df.index)

'''
Single request: https://cloud.iexapis.com/v1/stock/aapl/stats?token=XXXXX
Batch request (JSON): https://cloud.iexapis.com/v1/stock/market/batch?symbols=aapl,fb&types=stats&token=XXXXX
Batch request (CSV): https://cloud.iexapis.com/v1/stock/market/batch?symbols=aapl,fb&types=stats&format=csv&token=XXXXX
'''
with open(base_dir / 'config.json') as file:
    config = json.load(file)
iex_api_key = config['iex_api_key']
iex_api_endpoint = 'https://cloud.iexapis.com/v1'
iex_adapter = HTTPAdapter(max_retries=5)
iex_session = requests.Session()
iex_session.mount(iex_api_endpoint, iex_adapter) # Use `adapter` for all requests to endpoints that start with `iex_api_endpoint`

# Stock info and stock splits
all_stocks_info_path = base_dir / 'data/all_stocks_info.json'
if all_stocks_info_path.is_file():
    with open(all_stocks_info_path) as file:
        all_stocks_info = json.load(file)
        all_stocks_info_last_updated = datetime.fromtimestamp(os.path.getmtime(all_stocks_info_path)).replace(tzinfo=utc)
        # print(f'All stocks info last updated: {all_stocks_info_last_updated.astimezone(utc).strftime(time_format)}')
else:
    all_stocks_info = None
if all_stocks_info is None or all_stocks_info_last_updated < previous_close:
    try:
        # print('All stocks info is not up to date. Updating now.')
        # IEX Cloud batch limit: 100
        all_symbols_batches = split_into_batches(all_symbols, 100)
        responses = []
        for batch in all_symbols_batches:
            request_url = iex_api_endpoint + f'/stock/market/batch?symbols={batch_str_format(batch)}&types=stats,splits&token={iex_api_key}'
            responses.append(iex_session.get(request_url).json())
        all_stocks_info = {}
        # Merge dictionaries
        for response in responses:
            all_stocks_info.update(response)
        all_stocks_info = dict((k.lower(), v) for k, v in all_stocks_info.items()) # Make key names lowercase
        with open(all_stocks_info_path, 'w') as file:
            json.dump(all_stocks_info, file, indent=4)
    except ConnectionError as connection_error:
        print(connection_error)
stock_splits = {} # {'tsla': {'date': '2020-08-31', 'factor': 5}, ...}
for symbol in all_stocks_info:
    if len(all_stocks_info[symbol]['splits']) > 0:
        stock_splits[symbol] = {
            'date': pd.Timestamp(all_stocks_info[symbol]['splits'][0]['exDate']),
            'factor': all_stocks_info[symbol]['splits'][0]['toFactor'],
        }

# Fund/stock breakdown
fund_holdings_data = {'symbol': [], 'company': [], 'category': []}
for fund in funds:
    fund_holdings_data['count'] = []
    fund_holdings_data[fund] = []
for symbol in all_symbols:
    fund_holdings_data['symbol'].append(symbol)
    fund_holdings_data['company'].append(all_companies_df.loc[symbol, 'company'])
    count = 0
    for fund in funds:
        if symbol in funds[fund]['companies_df'].index:
            fund_holdings_data[fund].append(fund)
            count += 1
        else:
            fund_holdings_data[fund].append('')
    fund_holdings_data['count'].append(count)
    fund_holdings_data['category'].append('')
fund_holdings_df = pd.DataFrame.from_dict(fund_holdings_data, orient='columns').set_index('symbol').sort_values(by=['count', 'arkk', 'arkg', 'arkw', 'arkf', 'arkq', 'arkx'], ascending=False)
for stock in fund_holdings_df.index:
    count = fund_holdings_df.loc[stock, 'count']
    if fund_holdings_df.loc[stock, 'arkk'] != '':
        count -= 1
    if count == 1 and fund_holdings_df.loc[stock, 'arkf'] != '':
        fund_holdings_df.loc[stock, 'category'] = 'finance'
    elif count == 1 and fund_holdings_df.loc[stock, 'arkq'] != '':
        fund_holdings_df.loc[stock, 'category'] = 'autonomy'
    elif count == 1 and fund_holdings_df.loc[stock, 'arkg'] != '':
        fund_holdings_df.loc[stock, 'category'] = 'biotech'
    for label in category_labels:
        if stock in category_labels[label]:
            fund_holdings_df.loc[stock, 'category'] = label
# Replace all-caps company names with more readable versions
for symbol in fund_holdings_df.index:
    if symbol in all_stocks_info and all_stocks_info[symbol]['stats'] is not None and all_stocks_info[symbol]['stats']['companyName']:
        fund_holdings_df.loc[symbol, 'company'] = all_stocks_info[symbol]['stats']['companyName']
fund_holdings_df.index.names = ['Symbol']
fund_holdings = apply_style(fund_holdings_df.drop(['count'], axis=1))
# Used in separate back testing project
fund_holdings_csv_path = base_dir / 'output/ark_fund_holdings.csv'
os.makedirs(os.path.dirname(fund_holdings_csv_path), exist_ok=True)
fund_holdings_df.to_csv(fund_holdings_csv_path)

# Holdings by weight
for fund in funds:
    latest_df = map_csv_to_df(funds[fund]['csv_files'][-1])
    funds[fund]['holdings_by_weight_df'] = process_for_sort_by_weight(latest_df)
    if fund != 'arkk':
        funds[fund]['holdings_by_weight_df'] = process_unique_holdings(funds[fund]['holdings_by_weight_df'])
    funds[fund]['holdings_by_weight_df'] = process_percent_ownership(funds[fund]['holdings_by_weight_df'])
    funds[fund]['holdings_by_weight'] = apply_style(funds[fund]['holdings_by_weight_df'])

# Change in holdings and change in value
for fund in funds:
    funds[fund]['dates_from_data'] = [csv_path_to_date(csv_path) for csv_path in funds[fund]['csv_files']]
    funds[fund]['earliest_date_from_data'] = min(funds[fund]['dates_from_data'])
    funds[fund]['latest_date_from_data'] = max(funds[fund]['dates_from_data'])
    # Format date for indexing in `us_calendar.sessions_in_range`
    funds[fund]['dates_from_calendar'] = [session.date() for session in us_calendar.sessions_in_range(date_to_datetime_with_timezone(funds[fund]['earliest_date_from_data']), date_to_datetime_with_timezone(funds[fund]['latest_date_from_data']))]
    funds[fund]['missing_dates'] = [session for session in funds[fund]['dates_from_calendar'] if session not in funds[fund]['dates_from_data']]
    if len(funds[fund]['missing_dates']) > 0:
        print(f"{fund.upper()} holdings data is missing for the following dates: {funds[fund]['missing_dates']}")
    latest_df = map_csv_to_df(funds[fund]['csv_files'][-1])
    # Past two sessions
    if len(funds[fund]['csv_files']) >= 2:
        funds[fund]['change_in_holdings_past_two_sessions_df'] = process_for_change_in_holdings(map_csv_to_df(funds[fund]['csv_files'][-2]), latest_df, fund)
        funds[fund]['change_in_value_past_two_sessions_df'] = process_for_change_in_value(funds[fund]['change_in_holdings_past_two_sessions_df'])
        funds[fund]['change_in_holdings_past_two_sessions'] = apply_style(funds[fund]['change_in_holdings_past_two_sessions_df'])
        funds[fund]['change_in_value_past_two_sessions'] = apply_style(funds[fund]['change_in_value_past_two_sessions_df'].drop(columns=['contribution_abs', 'color']))
    # Past week
    one_week_back = funds[fund]['latest_date_from_data'] - relativedelta(days=7)
    if funds[fund]['earliest_date_from_data'] <= one_week_back:
        nearest_date = nearest(funds[fund]['dates_from_data'], one_week_back)
        index = funds[fund]['dates_from_data'].index(nearest_date)
        funds[fund]['change_in_holdings_past_week'] = apply_style(process_for_change_in_holdings(map_csv_to_df(funds[fund]['csv_files'][index]), latest_df, fund))
    # Past month
    one_month_back = funds[fund]['latest_date_from_data'] - relativedelta(months=1)
    if funds[fund]['earliest_date_from_data'] <= one_month_back:
        nearest_date = nearest(funds[fund]['dates_from_data'], one_month_back)
        index = funds[fund]['dates_from_data'].index(nearest_date)
        funds[fund]['change_in_holdings_past_month'] = apply_style(process_for_change_in_holdings(map_csv_to_df(funds[fund]['csv_files'][index]), latest_df, fund))
    # Past quarter
    one_quarter_back = funds[fund]['latest_date_from_data'] - relativedelta(months=3)
    if funds[fund]['earliest_date_from_data'] <= one_quarter_back:
        nearest_date = nearest(funds[fund]['dates_from_data'], one_quarter_back)
        index = funds[fund]['dates_from_data'].index(nearest_date)
        funds[fund]['change_in_holdings_past_quarter'] = apply_style(process_for_change_in_holdings(map_csv_to_df(funds[fund]['csv_files'][index]), latest_df, fund))
    # Past half year
    half_year_back = funds[fund]['latest_date_from_data'] - relativedelta(months=6)
    if funds[fund]['earliest_date_from_data'] <= half_year_back:
        nearest_date = nearest(funds[fund]['dates_from_data'], half_year_back)
        index = funds[fund]['dates_from_data'].index(nearest_date)
        funds[fund]['change_in_holdings_past_half_year'] = apply_style(process_for_change_in_holdings(map_csv_to_df(funds[fund]['csv_files'][index]), latest_df, fund))
    # Past year
    one_year_back = funds[fund]['latest_date_from_data'] - relativedelta(years=1)
    if funds[fund]['earliest_date_from_data'] <= one_year_back:
        nearest_date = nearest(funds[fund]['dates_from_data'], one_year_back)
        index = funds[fund]['dates_from_data'].index(nearest_date)
        funds[fund]['change_in_holdings_past_year'] = apply_style(process_for_change_in_holdings(map_csv_to_df(funds[fund]['csv_files'][index]), latest_df, fund))
