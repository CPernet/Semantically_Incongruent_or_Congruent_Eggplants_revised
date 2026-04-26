# Semantically Incongruent or Congruent Eggplants – Revised Analysis

A re-analysis of [Toffolo et al. (2022)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9540983/).

This repository contains Python and MATLAB code to re-analyze the EEG data
by regressing per-trial linguistic metrics on the EEG signal using
[EEGLAB](https://sccn.ucsd.edu/eeglab/) and the
[LIMO EEG](https://github.com/LIMO-EEG-Toolbox/limo_tools) plugin.

---

## Repository layout

```
python/
    word_frequency.py       – word lexical frequency (wordfreq)
    phonology.py            – phonological properties (CMU Pronouncing Dict)
    surprisal.py            – next-word prediction / surprisal (GPT-2)
    sentence_metrics.py     – sentence-level semantic similarity (BERT)
    process_stimuli.py      – CLI: run the full pipeline on a stimuli CSV
    requirements.txt        – Python dependencies
matlab/
    eeg_analysis.m          – EEGLAB / LIMO first- and second-level GLM
```

---

## Data sources

| Dataset | URL |
|---------|-----|
| Stimuli | https://datadryad.org/dataset/doi:10.5061/dryad.9ghx3ffkg |
| EEG (BIDS derivatives) | https://datadryad.org/dataset/doi:10.5061/dryad.6wwpzgmx4 |

---

## Python – linguistic metrics

### Installation

```bash
pip install -r python/requirements.txt
```

### Quick start

Prepare a CSV file with (at minimum) a `sentence` column and optionally a
`critical_word` column pointing to the sentence-final or otherwise critical
word.  Then run:

```bash
python python/process_stimuli.py stimuli.csv --output stimuli_metrics.csv
```

On first run the script will download GPT-2 (~500 MB) and
`bert-base-uncased` (~440 MB) model weights from HuggingFace and cache them
locally.

### Output columns

| Column | Description |
|--------|-------------|
| `cw_zipf_frequency` | Zipf-scale lexical frequency of the critical word |
| `cw_raw_frequency` | Raw corpus frequency of the critical word |
| `cw_n_letters` | Number of letters |
| `cw_phonemes` | ARPABET phoneme string |
| `cw_n_phonemes` | Number of phonemes |
| `cw_n_syllables` | Number of syllables |
| `cw_onset_phoneme` | Onset (first) phoneme |
| `cw_surprisal_bits` | GPT-2 surprisal of the critical word given its context (bits) |
| `sent_mean_surprisal` | Mean per-token surprisal across the whole sentence |
| `sent_perplexity` | Sentence perplexity (2^mean_surprisal) |
| `sent_n_tokens` | Number of sub-word tokens |
| `sent_context_target_sim` | Cosine similarity between sentence context and critical word (BERT) |

### Using the Python modules individually

```python
from python.word_frequency import get_zipf_frequency
from python.phonology import get_phonology_for_word
from python.surprisal import SurprisalModel
from python.sentence_metrics import SentenceMetrics

# Lexical frequency
print(get_zipf_frequency("eggplant"))   # e.g. 2.37

# Phonology
print(get_phonology_for_word("eggplant"))
# {'word': 'eggplant', 'phonemes': 'EH1 G P L AE2 N T', ...}

# Surprisal
model = SurprisalModel()
print(model.word_surprisal("She cooked the", "eggplant"))

# Sentence semantics
sm = SentenceMetrics()
print(sm.context_target_similarity("She cooked the", "eggplant"))
```

---

## MATLAB – EEG regression analysis

1. Download the BIDS derivative EEG data from Dryad (link above).
2. Install EEGLAB and the LIMO EEG plugin.
3. Edit the configuration block at the top of `matlab/eeg_analysis.m`
   (set `bids_root` and `metrics_csv`).
4. Run from the MATLAB command window:
   ```matlab
   eeg_analysis
   ```

The script runs a first-level GLM per subject (regressing the linguistic
metrics onto the EEG data) and a second-level robust group analysis with
LIMO.
