import os
import sys
from streamlit import cli as stcli

def run():
    _this_file = os.path.abspath(__file__)
    _this_directory = os.path.dirname(_this_file)

    file_path = os.path.join(_this_directory, 'omic_learn.py')

    HOME = os.path.expanduser("~")
    ST_PATH = os.path.join(HOME, ".streamlit")

    for folder in [ST_PATH]:
        if not os.path.isdir(folder):
            os.mkdir(folder)

    #Check if streamlit credentials exists
    ST_CREDENTIALS = os.path.join(ST_PATH, 'credentials.toml')
    if not os.path.isfile(ST_CREDENTIALS):
        with open(ST_CREDENTIALS, 'w') as file:
            file.write("[general]\n")
            file.write('\nemail = ""')

    print(f'Starting OmicLearn from {file_path}')

    args = ["streamlit", "run", file_path, "--global.developmentMode=false", "--browser.gatherUsageStats=False"]

    sys.argv = args

    sys.exit(stcli.main())
