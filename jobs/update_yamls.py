import pandas as pd
from datetime import datetime
import re
from copy import deepcopy

# model_pipeline

with open('../.github/workflows/model_pipeline.yml', 'r') as f:
    model_pipeline = f.read()

run_id_match = re.search(r'run-\d+', model_pipeline)
run_id = run_id_match.group(0) if run_id_match else "run-1"
run_n = int(re.search(r'\d+', run_id).group())
new_run_id = f"run-{run_n+1}"

model_version_match = re.search(r'--version \d+', model_pipeline)
model_version = model_version_match.group(0) if model_version_match else "--version 1"
version = int(re.search(r'\d+', model_version).group())
new_model_version = f"--version {version+1}"

model_pipeline = re.sub(r'--version \d+', new_model_version, model_pipeline)
model_pipeline = re.sub(r'run-\d+', new_run_id, model_pipeline)

with open('../.github/workflows/model_pipeline.yml', 'w') as f:
    f.write(model_pipeline)

# train

with open('../jobs/train.yml', 'r') as f:
    train = f.read()

if 'path:' in train and '@latest' in train:
    uri_path = train[train.index('path:'):train.index('@latest')].split(":")
    with open('../.github/workflows/data_pipeline.yml', 'r') as f:
        data_pipeline = f.read()
    ticker_string = data_pipeline[data_pipeline.index('ticker'): data_pipeline.index('.NS')]
    ticker = ticker_string[ticker_string.index(':'):].strip(":").strip(" ")
    new_uri_path = deepcopy(uri_path)
    new_uri_path[2] = ticker
    uri_path = ":".join(uri_path)
    new_uri_path = ":".join(new_uri_path)
    train = re.sub(re.escape(uri_path), new_uri_path, train)

train = re.sub(r'run-\d+', new_run_id, train)

with open('../jobs/train.yml', 'w') as f:
    f.write(train)

# deploy

with open('../jobs/deploy.yml', 'r') as f:
    deploy = f.read()

curr_model_match = re.search(r'GA_model:\d+', deploy)
if curr_model_match:
    curr_model = curr_model_match.group(0)
    curr_model_version = int(curr_model.split(':')[1]) + 1
    deploy = re.sub(r'GA_model:\d+', f'GA_model:{curr_model_version}', deploy)

with open('../jobs/deploy.yml', 'w') as f:
    f.write(deploy)