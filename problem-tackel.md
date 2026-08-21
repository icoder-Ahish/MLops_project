# MLOps Pipeline Troubleshooting & Problem Resolution Guide

This document captures all technical challenges, root causes, and solutions implemented across the Azure Machine Learning & GitHub Actions MLOps pipelines (**Data Pipeline**, **Model Training Pipeline**, and **Deployment Pipeline**).

---

## Table of Contents
1. [Missing Registered Azure ML Environment](#1-missing-registered-azure-ml-environment)
2. [MLflow & Azure ML Package Version Conflict (`ImportError`)](#2-mlflow--azure-ml-package-version-conflict-importerror)
3. [Azure ML Job Name Collision & API Version Incompatibility](#3-azure-ml-job-name-collision--api-version-incompatibility)
4. [GitHub Actions Workflow Output Corruption (`Invalid format 'Streaming...'`)](#4-github-actions-workflow-output-corruption-invalid-format-streaming)
5. [Single-Digit Versioning Failure in `update_yamls.py`](#5-single-digit-versioning-failure-in-update_yamlspy)
6. [Missing Output Directory in Training Script (`train.py`)](#6-missing-output-directory-in-training-script-trainpy)
7. [CLI Flag Error (`--all-traffic` on `deployment update`)](#7-cli-flag-error---all-traffic-on-deployment-update)
8. [Scoring Script Model Artifact Path & Input Payload Flexibility (`deployment.py`)](#8-scoring-script-model-artifact-path--input-payload-flexibility-deploymentpy)
9. [Data Asset Version Collision in `data_pipeline.yml`](#9-data-asset-version-collision-in-data_pipelineyml)
10. [Missing Azure Container Registry Resource (`Microsoft.ContainerRegistry/registries/... not found`)](#10-missing-azure-container-registry-resource-microsoftcontainerregistryregistries-not-found)

---

## 1. Missing Registered Azure ML Environment

### Symptom / Error
```text
ERROR: (UserError) No environment exists for name: stock-pricing, version: 4, label:
Code: UserError
Message: No environment exists for name: stock-pricing, version: 4, label:
```

### Root Cause
Job specification files (`jobs/train.yml` and `jobs/deploy.yml`) referenced hardcoded registered environment strings (e.g. `environment: azureml:stock-pricing:4`). Because version 4 was never created or registered in the target Azure ML Workspace, job submission failed.

### Solution / Overcome
Updated job specifications to use **Inline Environment Definitions**. Pointing directly to the Conda environment YAML files allows Azure ML to build and register the environment on the fly automatically during job submission.

**Updated `jobs/train.yml` and `jobs/deploy.yml`:**
```yaml
environment:
  image: mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest
  conda_file: ../config/training_env.yml
```

---

## 2. MLflow & Azure ML Package Version Conflict (`ImportError`)

### Symptom / Error
```text
ImportError: cannot import name 'Dataset' from 'mlflow.entities' (/azureml-envs/.../lib/python3.8/site-packages/mlflow/entities/__init__.py)
```

### Root Cause
1. **YAML Syntax Error**: `config/training_env.yml` had a space after `==` (`- mlflow== 1.26.1`), causing `pip` to misinterpret the constraint and pull `mlflow 2.x`.
2. **Version Mismatch**: `mlflow 2.x` files collided with `azureml-mlflow==1.42.0` (from MLflow 1.x era), resulting in broken site-packages where `mlflow.data` attempted to import `Dataset` from `mlflow.entities`.
3. **Conda vs Pip Dependency Collision**: `pandas` was specified as `pandas>=1.1,<1.2` in Conda dependencies but `pandas==1.5.3` in Pip dependencies.

### Solution / Overcome
1. Fixed YAML formatting by removing spaces in dependency definitions.
2. Upgraded `mlflow` to `>=2.4.0` (which includes `mlflow.entities.Dataset`) and unpinned `azureml-mlflow` to allow version compatibility.
3. Standardized `pandas` to version `1.5.3` across Conda and Pip blocks in `config/training_env.yml` and `config/deployment_env.yml`.

---

## 3. Azure ML Job Name Collision & API Version Incompatibility

### Symptom / Error
```text
ERROR: (UserError) A job was found, but it is not supported in this API version and cannot be accessed.
MessageParameters: "0": "ga-run-900"
```

### Root Cause
1. `name: ga-run-900` was hardcoded inside `jobs/train.yml`. Re-submitting pipeline runs attempted to reuse the exact same Azure ML job name, violating Azure ML's unique name constraint.
2. Azure ML attempted to inspect the existing stale job created under an older API schema, triggering a `JobNotSupported` error.

### Solution / Overcome
1. **Removed hardcoded job name** from `jobs/train.yml`.
2. **Dynamic Job Naming**: Updated `.github/workflows/model_pipeline.yml` to pass a dynamic `--name` flag using GitHub Actions run variables:
   ```bash
   AML_JOB_NAME="ga-run-${{ github.run_id }}-${{ github.run_attempt }}"

   az ml job create \
     --file train.yml \
     --name "$AML_JOB_NAME" \
     ...
   ```
Every run generates a guaranteed unique job name (e.g. `ga-run-12345678-1`), preventing collisions permanently.

---

## 4. GitHub Actions Workflow Output Corruption (`Invalid format 'Streaming...'`)

### Symptom / Error
```text
Error: Unable to process file command 'output' successfully.
Error: Invalid format 'Streaming user_logs/std_log.txt'
```

### Root Cause
In `.github/workflows/model_pipeline.yml`, `--stream` was combined with `--query name --output tsv` inside subshell execution `job_name="$(az ml job create --stream ...)"`. The CLI log streaming output (e.g. `Streaming user_logs/std_log.txt`) contaminated the captured `$job_name` variable, corrupting `echo "job_name=$job_name" >> "$GITHUB_OUTPUT"`.

### Solution / Overcome
Separated job submission and log streaming into two distinct execution steps:
```bash
# 1. Submit job cleanly without --stream to capture ONLY the job name
job_name="$(az ml job create \
  --file train.yml \
  --name "$AML_JOB_NAME" \
  --query name \
  --output tsv)"

echo "Submitted Azure ML job: $job_name"

# 2. Stream logs separately to stdout
az ml job stream --name "$job_name"

# 3. Write clean job name to GitHub step outputs
echo "job_name=$job_name" >> "$GITHUB_OUTPUT"
```

---

## 5. Single-Digit Versioning Failure in `update_yamls.py`

### Symptom / Error
Pipeline automation failed when run numbers or model versions exceeded single digits (`run-10`, `GA_model:10`).

### Root Cause
`jobs/update_yamls.py` used single-digit regular expressions such as `re.findall('run-[0-9]', ...)` and `int(re.findall('[0-9]', run_id)[0])`. When `run-9` incremented to `run-10`, `[0-9]` extracted only `'1'`, resetting `run-10` back to `run-2` or causing index errors.

### Solution / Overcome
Refactored `jobs/update_yamls.py` to use `\d+` regexes to safely match and increment multi-digit integers:
```python
run_id_match = re.search(r'run-\d+', model_pipeline)
run_n = int(re.search(r'\d+', run_id).group())
new_run_id = f"run-{run_n+1}"
```

---

## 6. Missing Output Directory in Training Script (`train.py`)

### Symptom / Error
```text
FileNotFoundError: [Errno 2] No such file or directory: './outputs/scaler.pkl'
```

### Root Cause
`jobs/train.py` attempted to write binary artifacts (`scaler.pkl` and `model.pth`) directly to `./outputs/` without ensuring the output directory existed on fresh compute node working directories.

### Solution / Overcome
Added explicit directory creation before saving artifacts in `jobs/train.py`:
```python
os.makedirs('./outputs', exist_ok=True)
pickle.dump(scalerObj, open('./outputs/scaler.pkl', 'wb'))
torch.save(trainedModel.state_dict(), './outputs/model.pth')
```

---

## 7. CLI Flag Error (`--all-traffic` on `deployment update`)

### Symptom / Error
```text
ERROR: unrecognized arguments: --all-traffic
```

### Root Cause
In Azure ML CLI v2, `--all-traffic` is valid for `az ml online-deployment create`, but **invalid** for `az ml online-deployment update`.

### Solution / Overcome
Updated `.github/workflows/deployment_pipeline.yml` to remove `--all-traffic` from `online-deployment update` and route traffic explicitly using `az ml online-endpoint update`:
```yaml
az ml online-deployment create -f deploy.yml --all-traffic || \
  az ml online-deployment update -f deploy.yml
az ml online-endpoint update --name ga-deployment --traffic green=100 || true
```

---

## 8. Scoring Script Model Artifact Path & Input Payload Flexibility (`deployment.py`)

### Symptom / Error
Potential crashes when mounting model artifacts inside `AZUREML_MODEL_DIR` or when receiving diverse JSON payloads during HTTP scoring requests.

### Root Cause
1. Rigid path lookup (`AZUREML_MODEL_DIR/outputs/scaler.pkl`) failed if Azure ML mounted files directly at `AZUREML_MODEL_DIR`.
2. Rigid input dictionary parsing failed if client requests sent raw lists or JSON wrappers.

### Solution / Overcome
Updated `jobs/deployment.py`:
1. **Fallback Path Loading**: Checks both `AZUREML_MODEL_DIR/outputs/` and `AZUREML_MODEL_DIR/` for model and scaler files.
2. **Payload Parsing**: Handles list inputs, dict inputs with `"data"` keys, and dictionary value arrays cleanly:
   ```python
   def run(raw_data):
       try:
           data_json = json.loads(raw_data)
           if isinstance(data_json, dict) and "data" in data_json:
               values = data_json["data"]
           elif isinstance(data_json, dict):
               values = list(data_json.values())
           else:
               values = data_json
           
           values = np.array(values).astype(float)
           pred_data = datamod.predict_dataloader(data=pd.DataFrame(values, columns=['Close']))
           result, _ = mod(pred_data)
           result = scaler.inverse_transform(result.detach().numpy())
           return {"forecast": result.tolist()}
       except Exception as e:
           return {"error": str(e)}
   ```

---

## 9. Data Asset Version Collision in `data_pipeline.yml`

### Symptom / Error
```text
ERROR: (UserError) A data version with this name and version already exists. If you are trying to create a new data version, use a different name or version. If you are trying to update an existing data version, the existing asset's data uri cannot be changed. Only tags, description, and isArchived can be updated.
Code: UserError
Message: A data version with this name and version already exists. If you are trying to create a new data version, use a different name or version.
```

### Root Cause
`jobs/data_download.py` generated data asset version strings using only `date.today()` (e.g. `'20260821'`). When `.github/workflows/data_pipeline.yml` executed multiple times on the same date (or when re-running after prior uploads), `az ml data create -f jobs/data_upload.yml` attempted to register an already existing version string, which Azure Machine Learning prohibits for data assets.

### Solution / Overcome
Updated `save_to_data_upload` in `jobs/data_download.py` to produce granular timestamp-based version numbers using `datetime.now().strftime("%Y%m%d%H%M%S")`:
```python
from datetime import datetime, date

# Generate timestamp-based version string (e.g. 20260821074500)
version = datetime.now().strftime("%Y%m%d%H%M%S")
```
This guarantees unique dataset version identifiers across multiple pipeline runs on the same day, preventing version collisions during `az ml data create`.

---

## 10. Missing Azure Container Registry Resource (`Microsoft.ContainerRegistry/registries/... not found`)

### Symptom / Error
```text
"message": "Unable to get image details : Unable to fetch workspace resources: The Resource 'Microsoft.ContainerRegistry/registries/d8397a9a40d544c093cc0871f20bd99a' under resource group 'ashis-mlop' was not found. For more details please go to https://aka.ms/ARMResourceNotFoundFix."
```

### Root Cause
When an Azure Machine Learning workspace is initialized, it binds to a specific Azure Container Registry (ACR) to store environment images built for compute clusters. If that ACR instance was deleted from Azure (e.g. during subscription cleanup or trial resource deletion), the workspace's internal ARM metadata (`containerRegistry` reference) points to a non-existent resource ID, breaking environment image builds during model training or deployment.

### Solution / Overcome
Recreate the Azure ML workspace so Azure automatically provisions and binds a fresh, fully connected ACR, Storage Account, and Key Vault:

```bash
# 1. Delete broken workspace reference
az ml workspace delete --name mlopssecond --resource-group ashis-mlop --yes

# 2. Recreate fresh workspace (auto-provisions linked ACR & dependencies)
az ml workspace create --name mlopssecond --resource-group ashis-mlop --location centralindia
```


