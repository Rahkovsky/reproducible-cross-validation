# reproducible-cross-validation
The repo measured the variation of Random Forest (Catboost) results with respected to changed in Catboost
random seed, random data shuffle, and cross-validation random seed.

See discussion in [the post](https://ilya0.substack.com/p/reproducibility-and-hidden-choices)

# Run

### If you need to install python3.11
```bash
sudo apt update
sudo apt install -y software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install -y python3.11 python3.11-venv python3.11-dev
python3.11 --version
```

### Install virtual environment (Linux)
```bash
python3.11 -m venv venv
source venv/bin/activate
pip install uv
uv pip install -r requirements.txt 
```

### Run the code
```bash
# estimate models

python3.11 random_seed_experiments.py --randomize-cb-rs --randomize-cv-rs --randomize-shuffle-rs --g-count 100 --output-file cb_cv_shuffle__random__g_100.csv
python3.11 random_seed_experiments.py --randomize-cb-rs --g-count 100 --output-file cb_random__g_100.csv
python3.11 random_seed_experiments.py --randomize-cb-rs --randomize-shuffle-rs --g-count 100 --output-file cb_shuffle__random__g_100.csv
python3.11 random_seed_experiments.py --randomize-cb-rs --randomize-cv-rs --g-count 100 --output-file cv_cb__random__g_100.csv
python3.11 random_seed_experiments.py --randomize-cv-rs --g-count 100 --output-file cv_random__g_100.csv
python3.11 random_seed_experiments.py --randomize-cv-rs --randomize-shuffle-rs --g-count 100 --output-file cv_shuffle__random__g_100.csv
python3.11 random_seed_experiments.py --randomize-shuffle-rs --g-count 100 --output-file shuffle_random__g_100.csv


# visualization
python3.11 visualization.py
```