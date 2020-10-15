'''
This script checks ARK's website for new fund holdings data and downloads it when it is published.
Then it runs the notebook and exports it as an HTML file.

Run `crontab -e` and add the following (with correct paths) to check for new data every 15 minutes (starting at minute 2) and export the notebook when new data is downloaded.
PYTHON_PATH=/path/to/python
ARK_FUND_ANALYSIS=/path/to/ark_fund_analysis
COMMAND="cd $ARK_FUND_ANALYSIS && $PYTHON_PATH cron_export_notebook.py"
2-59/15 * * * * (eval $COMMAND) 2>> $ARK_FUND_ANALYSIS/output/cron_stderr.log 1> $ARK_FUND_ANALYSIS/output/cron_stdout.log
'''

from pathlib import Path
from datetime import datetime
import pytz
import subprocess
import os
import json

from automation import export_notebook, download_fund_holdings_data, download_fund_daily_price_data

utc = pytz.utc
est = pytz.timezone('US/Eastern')
time_format = '%Y-%m-%d %H:%M:%S %z'
base_dir = Path(__file__).parent.absolute()
exported_notebook_path = base_dir / 'output/ark_fund_analysis.html'

with open(base_dir / 'config.json') as file:
    config = json.load(file)

fund_data_updated = download_fund_holdings_data()
fund_data_updated = fund_data_updated.astimezone(utc).strftime(time_format)
if os.path.isfile(exported_notebook_path):
    notebook_updated = datetime.fromtimestamp(os.path.getmtime(exported_notebook_path)).replace(tzinfo=utc).strftime(time_format)
    print(f'Notebook updated:            {notebook_updated}')
    print(f'ARK fund data updated:       {fund_data_updated}')
else:
    notebook_updated = None

if notebook_updated is None or fund_data_updated > notebook_updated:
    download_fund_daily_price_data() # Calling this here rather than in the notebook to capture stderr in case of error
    subprocess.run(f"cd '{base_dir}' && {config['git_path']} pull", shell=True, check=True) # Get the latest version of the notebook
    export_notebook()
