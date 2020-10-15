'''
This script checks ARK's website for new fund holdings data and downloads it when it is published.

Run `crontab -e` and add the following (with correct paths) to check for new data every hour at minute 2.
PYTHON_PATH=/path/to/python
ARK_FUND_ANALYSIS=/path/to/ark_fund_analysis
2 * * * * cd $ARK_FUND_ANALYSIS && $PYTHON_PATH cron_update_data.py 2>> $ARK_FUND_ANALYSIS/output/cron_stderr.log 1> $ARK_FUND_ANALYSIS/output/cron_stdout.log
'''

from automation import download_fund_holdings_data
download_fund_holdings_data()
