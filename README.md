# Client Retention Analytics

## Churn Prediction

## Project Installation

### Local

Create virtual environment

```bash
python -m venv .venv
```

Activate venv

```bash
# Windows
.venv\Scripts\activate

# Linux
source .venv/Scripts/activate
```

Install requirements.txt

```
pip install -r requirements.txt
```

Execute main

```
python -u main.py
```

### Virtual Machine

First! Check if python and venv are installed
```bash
python3 --version
sudo apt install -y python3.12-venv
```

THEN, clone the repo
```bash
git clone https://github.com/JessicaAquino/client-retention-analytics.git
```

Inside the folder...
```bash
cd client-retention-analytics
```

Create and activate the environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install requirements (for the VM)
```bash
pip install -r vm_requirements.txt
```

Now! We execute the main.py as a background process.
```bash
# Cool way
nohup python3 -u main.py > vm_execution.log 2>&1 &
```

Finish! Thanks for reading.