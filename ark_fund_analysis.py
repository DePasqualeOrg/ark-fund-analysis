import pandas as pd
import os
import requests
from requests.adapters import HTTPAdapter

from datetime import datetime
from dateutil import tz
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

base_dir = Path(__file__).parent.absolute()

utc = tz.tzutc()
now_utc = datetime.utcnow().replace(tzinfo=utc) # Add time zone
timestamp_now = pd.Timestamp(now_utc)
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
    'market value($)': 'Market value',
    'market value($)_x': 'Market value 1',
    'market value($)_y': 'Market value 2',
    'weight(%)': 'Weight',
    'weight(%)_x': 'Weight 1',
    'weight(%)_y': 'Weight 2',
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
    column_names['market value($)']: dollars_int,
    column_names['weight(%)']: perc_two_dec,
    column_names['shares']: int_comma_sep,
    column_names['market value($)']: dollars_int,
    column_names['shares_x']: int_comma_sep,
    column_names['market value($)_x']: dollars_int,
    column_names['weight(%)_x']: perc_two_dec,
    column_names['shares_y']: int_comma_sep,
    column_names['market value($)_y']: dollars_int,
    column_names['weight(%)_y']: perc_two_dec,
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

def process_df(df):
    df_copy = df.copy()
    # Use lowercase symbols in code and uppercase for display
    for index in df_copy.index:
        if not pd.isnull(df_copy.loc[index, 'ticker']):
            # Remove endings and whitespace from symbols
            df_copy.loc[index, 'ticker'] = df_copy.loc[index, 'ticker'].strip().lower() # Remove trailing and leading whitespace from symbols, and make lowercase
            for ending in [' uw', ' uq', ' un', ' u']:
                if df_copy.loc[index, 'ticker'].endswith(ending):
                    df_copy.loc[index, 'ticker'] = df_copy.loc[index, 'ticker'].rstrip(ending) # Remove ending from end of string
            # Custom fixes
            if df_copy.loc[index, 'ticker'] == 'dsy' and df_copy.loc[index, 'company'] == 'DISCOVERY LTD':
                df_copy.loc[index, 'ticker'] = 'dsy.jo' # Disambiguate Dassault Systems and Discovery Limited (ARK's data uses same symbol DSY for both)
            if df_copy.loc[index, 'ticker'] == 'tcs li':
                df_copy.loc[index, 'ticker'] = 'tcs.li'
            if df_copy.loc[index, 'ticker'] == 'dsy fp':
                df_copy.loc[index, 'ticker'] = 'dsy.fp'
            # Check for other cases not yet accounted for
            if ' ' in df_copy.loc[index, 'ticker']:
                symbols_with_spaces.append(df_copy.loc[index, 'ticker'])
    # Aggregate multiple rows with same asset (e.g. Japanese yen)
    aggregate_functions = {'date': 'first', 'fund': 'first', 'company': 'first', 'ticker': 'first', 'cusip': 'first', 'shares': 'sum', 'market value($)': 'sum', 'weight(%)': 'sum'}
    df_copy = df_copy.groupby(df_copy['cusip'], as_index=False).aggregate(aggregate_functions).sort_values(by=['weight(%)'], ascending=False).reset_index(drop=True)
    df_copy.dropna(subset=['fund'], inplace=True)
    df_copy['date'] = pd.to_datetime(df_copy['date'], format='%m/%d/%Y').dt.date # Convert to datetime object and display only date without time
    total_value = df_copy['market value($)'].sum()
    df_copy['share_price'] = df_copy['market value($)'] / df_copy['shares']
    df_copy['weight(%)'] = df_copy['market value($)'] / total_value # Recalculate to remove rounding errors
    return df_copy

def split_into_batches(full_list, batch_size):
    batches = []
    for i in range(0, math.ceil(len(full_list) / batch_size)):
        batches.append(full_list[(i * batch_size):((i + 1) * batch_size)])
    return batches

def add_totals(df):
    df_copy = df.copy()
    if 'weight(%)_x' in df_copy.columns:
        df_copy.loc['Total', 'weight(%)_x'] = df_copy['weight(%)_x'].sum()
        df_copy.loc['Total', 'market value($)_x'] = df_copy['market value($)_x'].sum()
        df_copy.loc['Total', 'weight(%)_y'] = df_copy['weight(%)_y'].sum()
        df_copy.loc['Total', 'market value($)_y'] = df_copy['market value($)_y'].sum()
    elif 'weight(%)' in df_copy.columns:
        df_copy.loc['Total', 'weight(%)'] = df_copy['weight(%)'].sum()
        df_copy.loc['Total', 'market value($)'] = df_copy['market value($)'].sum()
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
    for i in merged.index:
        if pd.notna(merged.loc[i, 'company_y']):
            merged.loc[i, 'company_x'] = merged.loc[i, 'company_y'] # Using `company_x` to display, so need to copy more recent name
        if merged.loc[i, 'ticker'] in stock_splits:
            if stock_splits[merged.loc[i, 'ticker']]['date'].date() >= switchover:
                # Move ahead one day to adjust for ARK's dating of CSV files (see note above)
                split_date_adjusted = us_calendar.next_open(stock_splits[merged.loc[i, 'ticker']]['date'].date() + pd.DateOffset(1)).date()
            else:
                split_date_adjusted = stock_splits[merged.loc[i, 'ticker']]['date'].date()
            if split_date_adjusted > start_date and split_date_adjusted <= end_date:
                merged.loc[i, 'split_factor'] = stock_splits[merged.loc[i, 'ticker']]['factor']
                merged.loc[i, 'shares_y'] /= merged.loc[i, 'split_factor']
                merged.loc[i, 'share_price_y'] *= merged.loc[i, 'split_factor']
    merged['change_in_share_price'] = (merged['share_price_y'] - merged['share_price_x']) / merged['share_price_x']
    merged['change_in_value'] = (merged['market value($)_y'] - merged['market value($)_x']) / merged['market value($)_x']
    merged['change_in_weight'] = merged['weight(%)_y'] - merged['weight(%)_x']
    merged['relative_change_in_weight'] = merged['change_in_weight'] / merged['weight(%)_x']
    merged['change_in_shares'] = merged['shares_y'] - merged['shares_x']
    merged['percent_change_in_shares'] = merged['change_in_shares'] / merged['shares_x']
    for i in merged.index:
        if pd.isna(merged.loc[i, 'weight(%)_x']):
            merged.loc[i, 'sort_rank'] = 1000000
        elif pd.isna(merged.loc[i, 'weight(%)_y']):
            merged.loc[i, 'sort_rank'] = -1000000
        else:
            merged.loc[i, 'sort_rank'] = merged.loc[i, 'percent_change_in_shares']
    merged = add_totals(merged)
    merged = merged.sort_values(by=['sort_rank', 'percent_change_in_shares'], ascending=False).reset_index(drop=True)
    merged.rename(index={merged.index[-1]: 'Total'}, inplace=True) # Need to rename last row of index after resetting index
    if 'split_factor' in merged.columns:
        merged = merged.drop(['split_factor'], axis=1)
    merged.loc['Total', 'change_in_value'] = (merged.loc['Total', 'market value($)_y'] - merged.loc['Total', 'market value($)_x']) / merged.loc['Total', 'market value($)_x']
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
    merged = merged.drop(['fund', 'cusip', 'sort_rank', 'change_in_shares', 'market value($)_x', 'market value($)_y', 'change_in_weight', 'relative_change_in_weight', 'shares_x', 'shares_y', 'share_price_x', 'share_price_y', 'company_y'], axis=1)
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
        for i in df.index:
            if pd.isna(df.loc[i, 'ticker']):
                df.loc[i, 'ticker'] = ''
    if df.index.name == 'Symbol':
        df.index = df.index.map(str.upper)
    styled = df.rename(columns=column_names).style.format(number_formats).applymap(negative_red_hide_empty).set_properties(**{'background-color': ''})
    if 'Total' in df.index:
        styled = styled.apply(total_row_bold)
    if 'unique_weight' in df.columns:
        styled = styled.hide_columns(['unique_weight'])
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
    change_in_value_df = change_in_holdings_df[['company_x', 'ticker', 'weight(%)_x', 'change_in_share_price', 'change_in_value']].drop(index='Total').copy()
    change_in_value_df['contribution'] = change_in_value_df['weight(%)_x'] * change_in_value_df['change_in_share_price']
    change_in_value_df['contribution_abs'] = abs(change_in_value_df['contribution'])
    for i in change_in_value_df.index:
        change_in_share_price = change_in_value_df.loc[i, 'change_in_share_price']
        cisp_clamp_abs = abs(clamp(change_in_share_price, -clamp_threshold, clamp_threshold))
        if change_in_share_price >= 0:
            color = green
            color.luminance = max_lum - (lum_diff * (cisp_clamp_abs / clamp_threshold))
        else:
            color = red
            color.luminance = max_lum - (lum_diff * (cisp_clamp_abs / clamp_threshold))
        change_in_value_df.loc[i, 'color'] = color.hex
    change_in_value_df = change_in_value_df.sort_values(by=['contribution'], ascending=False).reset_index(drop=True)
    change_in_value_df = change_in_value_df.reset_index(drop=True) # Sort all rows except last row (totals)
    change_in_value_df.loc['Total', 'weight(%)_x'] = change_in_value_df['weight(%)_x'].sum()
    change_in_value_df.loc['Total', 'contribution'] = change_in_value_df['contribution'].sum()
    change_in_value_df.loc['Total', 'change_in_value'] = change_in_holdings_df.loc['Total', 'change_in_value']
    change_in_value_df.loc['Total', 'change_in_share_price'] = change_in_holdings_df.loc['Total', 'change_in_share_price']
    return change_in_value_df

def nearest(items, pivot):
    return min(items, key=lambda x: abs(x - pivot))

def batch_str_format(symbols):
    return ','.join(symbols)

def process_for_sort_by_weight(df):
    df_copy = df.sort_values(by=['weight(%)'], ascending=False)
    df_copy = add_totals(df_copy)
    df_copy = df_copy.drop(['fund', 'date', 'cusip', 'share_price'], axis=1)
    return df_copy

def process_percent_ownership(df):
    for index in df.index:
        ticker = df.loc[index, 'ticker']
        # IEX Cloud
        if pd.notna(df.loc[index, 'ticker']) and ticker in all_stocks_info and all_stocks_info[ticker]['stats'] is not None and all_stocks_info[ticker]['stats']['sharesOutstanding'] != 0 and all_stocks_info[ticker]['stats']['marketcap'] is not None:
            df.loc[index, 'market_cap'] = all_stocks_info[ticker]['stats']['marketcap'] / 1000000000
            df.loc[index, 'shares_outstanding'] = all_stocks_info[ticker]['stats']['sharesOutstanding']
            df.loc[index, 'percent_ownership'] = df.loc[index, 'shares'] / df.loc[index, 'shares_outstanding']
            df.loc[index, 'total_ownership_contribution'] = df.loc[index, 'weight(%)'] * df.loc[index, 'percent_ownership']
            df.loc[index, 'pe_ratio'] = all_stocks_info[ticker]['stats']['peRatio']
    df.loc['Total', 'total_ownership_contribution'] = df['total_ownership_contribution'].sum()
    df.loc['Total', 'percent_ownership'] = df.loc['Total', 'total_ownership_contribution']
    df = df.drop(['shares_outstanding', 'total_ownership_contribution'], axis=1)
    return df

def create_share_changes_df(fund, symbol):
    previous_n_shares = None
    df_data = {'date': [], 'shares': [], 'perc_shares_change': [], 'n_shares_change': []}
    for i, df in enumerate(funds[fund]['dfs']):
        df = df.set_index('ticker')
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
            df_data['shares'].append(None)
            df_data['perc_shares_change'].append(None)
            df_data['n_shares_change'].append(None)
    stock_purchases_df = pd.DataFrame.from_dict(df_data, orient='columns').set_index('date')
    return stock_purchases_df

def process_unique_holdings(df):
    arkk_comparison_df = funds['arkk']['holdings_by_weight_df'].set_index('ticker')
    unique_holdings = 0
    comparison_df = df.set_index('ticker')
    comparison_df = comparison_df[comparison_df['company'].notna()] # Only keep rows with company name (removes 'Total' row)
    for symbol in comparison_df.index:
        if pd.notna(symbol) and symbol not in arkk_comparison_df.index:
            unique_holding_weight = comparison_df.loc[symbol, 'weight(%)']
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
    if len(funds[fund]['dfs']) >= 2:
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
            weight = df.loc[symbol, 'weight(%)']
            share_changes_df = create_share_changes_df(fund, symbol).tail(days_to_display)
            title = f'{fund.upper()}: {company} ({symbol.upper()}), weight: {round(weight * 100, 2)}%'
            ax = fig.add_subplot(spec[row, column])
            share_changes_df.plot(kind='bar', width=0.9, ax=ax, y='n_shares_change', title=title, legend=False).xaxis.label.set_visible(False)
            max_shares = share_changes_df['shares'].max()
            ax.set_ylim([-max_shares, max_shares]) # Use maximum number of shares held during time period as +/- limits for y axis
            # ax.figure.autofmt_xdate() # Removes date labels from x axis except in last row
            ax.ticklabel_format(scilimits=(0, 0), axis='y') # Always use scientific notation on y axis (more compact)
        plt.show()

def plot_share_price_and_estimated_capital_flows(fund, start_date=None):
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
    for i, df in enumerate(funds[fund]['dfs']):
        if start_date is None:
            start_date = funds[fund]['earliest_date_from_data']
        if df['date'].dropna().iloc[0] >= start_date:
            if i == 0:
                data['date'].append(df['date'].dropna().iloc[0])
                data['change_in_share_price'].append(None)
                data['change_in_value'].append(None)
                data['estimated_change_in_share_price'].append(None)
                data['estimated_capital_flows'].append(None)
            elif i > 0 and i < len(funds[fund]['dfs']):
                change_in_holdings_df = process_for_change_in_holdings(funds[fund]['dfs'][i - 1], funds[fund]['dfs'][i], fund)
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
            data['share_price_cumulative'].append(1)
            data['estimated_capital_flows_cumulative'].append(1)
        else:
            data['share_price_cumulative'].append(data['share_price_cumulative'][i - 1] * (1 + data['change_in_share_price'][i]))
            if pd.notna(data['estimated_capital_flows'][i]):
                data['estimated_capital_flows_cumulative'].append(data['estimated_capital_flows_cumulative'][i - 1] * (1 + data['estimated_capital_flows'][i]))
            else:
                data['estimated_capital_flows_cumulative'].append(data['estimated_capital_flows_cumulative'][i - 1])
    df = pd.DataFrame.from_dict(data)
    df['difference'] = df['change_in_share_price'] - df['estimated_change_in_share_price']
    df = df.set_index('date')
    # df[['difference']].plot.line(y='difference', figsize=(20, 5)) # Check if estimated change in share price based on data is within ordinary margin of error
    funds[fund]['estimated_capital_flows_and_share_price_df'] = df.copy()
    df[['share_price_cumulative', 'estimated_capital_flows_cumulative']].rename(columns={'share_price_cumulative': 'Share price', 'estimated_capital_flows_cumulative': 'Estimated capital flows'}).plot.line(title=f'{fund.upper()}', figsize=(20, 5))
    plt.legend(loc='upper left')
    plt.show()

# =====

fund_holdings_path = base_dir / 'data/ark_fund_holdings'
files = list(fund_holdings_path.glob('*.csv'))

for fund in funds:
    funds[fund]['files'] = sorted([path for path in files if fund in str(path)])
    funds[fund]['dfs'] = []
    for file in funds[fund]['files']:
        funds[fund]['dfs'].append(process_df(pd.read_csv(file)))
    companies_data = {'symbol': [], 'company': []}
    for i in funds[fund]['dfs'][-1].index:
        ticker = funds[fund]['dfs'][-1].loc[i, 'ticker']
        if not pd.isna(ticker):
            companies_data['symbol'].append(ticker)
            companies_data['company'].append(funds[fund]['dfs'][-1].loc[i, 'company'])
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

for fund in funds:
    funds[fund]['holdings_by_weight_df'] = process_for_sort_by_weight(funds[fund]['dfs'][-1])
    if fund != 'arkk':
        funds[fund]['holdings_by_weight_df'] = process_unique_holdings(funds[fund]['holdings_by_weight_df'])
    funds[fund]['holdings_by_weight_df'] = process_percent_ownership(funds[fund]['holdings_by_weight_df'])
    funds[fund]['holdings_by_weight'] = apply_style(funds[fund]['holdings_by_weight_df'])

# Change in holdings and change in value
for fund in funds:
    funds[fund]['dates_from_data'] = [df.loc[0, 'date'] for df in funds[fund]['dfs']]
    funds[fund]['earliest_date_from_data'] = min(funds[fund]['dates_from_data'])
    funds[fund]['latest_date_from_data'] = max(funds[fund]['dates_from_data'])
    # Format date for indexing in `us_calendar.sessions_in_range`
    funds[fund]['dates_from_calendar'] = [session.date() for session in us_calendar.sessions_in_range(date_to_datetime_with_timezone(funds[fund]['earliest_date_from_data']), date_to_datetime_with_timezone(funds[fund]['latest_date_from_data']))]
    funds[fund]['missing_dates'] = [session for session in funds[fund]['dates_from_calendar'] if session not in funds[fund]['dates_from_data']]
    if len(funds[fund]['missing_dates']) > 0:
        print(f"{fund.upper()} holdings data is missing for the following dates: {funds[fund]['missing_dates']}")
    # Past two sessions
    if len(funds[fund]['dfs']) >= 2:
        funds[fund]['change_in_holdings_past_two_sessions_df'] = process_for_change_in_holdings(funds[fund]['dfs'][-2], funds[fund]['dfs'][-1], fund)
        funds[fund]['change_in_value_past_two_sessions_df'] = process_for_change_in_value(funds[fund]['change_in_holdings_past_two_sessions_df'])
        funds[fund]['change_in_holdings_past_two_sessions'] = apply_style(funds[fund]['change_in_holdings_past_two_sessions_df'])
        funds[fund]['change_in_value_past_two_sessions'] = apply_style(funds[fund]['change_in_value_past_two_sessions_df'].drop(columns=['contribution_abs', 'color']))
    # Past week
    one_week_back = funds[fund]['latest_date_from_data'] - relativedelta(days=7)
    if funds[fund]['earliest_date_from_data'] <= one_week_back:
        nearest_date = nearest(funds[fund]['dates_from_data'], one_week_back)
        index = funds[fund]['dates_from_data'].index(nearest_date)
        funds[fund]['change_in_holdings_past_week'] = apply_style(process_for_change_in_holdings(funds[fund]['dfs'][index], funds[fund]['dfs'][-1], fund))
    # Past month
    one_month_back = funds[fund]['latest_date_from_data'] - relativedelta(months=1)
    if funds[fund]['earliest_date_from_data'] <= one_month_back:
        nearest_date = nearest(funds[fund]['dates_from_data'], one_month_back)
        index = funds[fund]['dates_from_data'].index(nearest_date)
        funds[fund]['change_in_holdings_past_month'] = apply_style(process_for_change_in_holdings(funds[fund]['dfs'][index], funds[fund]['dfs'][-1], fund))
    # Past quarter
    one_quarter_back = funds[fund]['latest_date_from_data'] - relativedelta(months=3)
    if funds[fund]['earliest_date_from_data'] <= one_quarter_back:
        nearest_date = nearest(funds[fund]['dates_from_data'], one_quarter_back)
        index = funds[fund]['dates_from_data'].index(nearest_date)
        funds[fund]['change_in_holdings_past_quarter'] = apply_style(process_for_change_in_holdings(funds[fund]['dfs'][index], funds[fund]['dfs'][-1], fund))
    # Past half year
    half_year_back = funds[fund]['latest_date_from_data'] - relativedelta(months=6)
    if funds[fund]['earliest_date_from_data'] <= half_year_back:
        nearest_date = nearest(funds[fund]['dates_from_data'], half_year_back)
        index = funds[fund]['dates_from_data'].index(nearest_date)
        funds[fund]['change_in_holdings_past_half_year'] = apply_style(process_for_change_in_holdings(funds[fund]['dfs'][index], funds[fund]['dfs'][-1], fund))
    # Past year
    one_year_back = funds[fund]['latest_date_from_data'] - relativedelta(years=1)
    if funds[fund]['earliest_date_from_data'] <= one_year_back:
        nearest_date = nearest(funds[fund]['dates_from_data'], one_year_back)
        index = funds[fund]['dates_from_data'].index(nearest_date)
        funds[fund]['change_in_holdings_past_year'] = apply_style(process_for_change_in_holdings(funds[fund]['dfs'][index], funds[fund]['dfs'][-1], fund))
# arkk_change_in_holdings_oldest_newest = apply_style(process_for_change_in_holdings(funds['arkk']['dfs'][0], funds['arkk']['dfs'][-1], fund))
