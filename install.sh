pip install -r requirements.txt
cp -n config_template.json config.json # If file exists, don't overwrite

cat << END_OF_MESSAGE
$(tput setaf 2)
Complete the following steps to finish the installation process:
- Enter the absolute paths for jupyter, git, and python in \`config.json\`:
  - $(which jupyter)
  - $(which git)
  - $(which python)
- Set up free API keys at the following sites and enter them in \`config.json\`:
  - https://iexcloud.io
  - https://www.alphavantage.co
- Set up a cron job for \`cron_update_data.py\` or \`cron_export_notebook.py\` if desired. Examples are provided in the comment at the top of these files.
END_OF_MESSAGE
