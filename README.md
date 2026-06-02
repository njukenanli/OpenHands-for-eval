# Benchmarking OpenHands with any LLM

## Prepare venv

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install uv
uv sync --dev --active
```

## Azure OpenAI identity login

```bash
source venv/bin/activate
python -m pip install openai azure-identity-broker --upgrade
```

Prepare your Python script to get function `get_azure_ad_token_provider`, say `cloudgpt_aoai.py`. Make sure `from cloudgpt_aoai import get_openai_token_provider` at `benchmarks/swebench/run_infer.py` works from this directory. The runner refreshes an Azure AD token per instance using `get_openai_token_provider()()`.

## Rollout
Modify config/default.yaml for your setting.

```bash
source venv/bin/activate
FORCE_BUILD=1 python main.py \
    --config config/default.yaml \
    --run-id debug \
    --dataset princeton-nlp/SWE-bench_Verified \
    --split test 

FORCE_BUILD=1 nohup python main.py \
    --config config/ds4pro.yaml \
    --run-id multilang \
    --dataset datasets/multilang.jsonl \
    > log-ds4pro.out 2>&1 &
```

Outputs are written under `logs/<model>/<run-id>/`.

