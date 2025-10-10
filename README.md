# LASQ
## Towards Human-Like Grading: A Unified LLM-Enhanced Framework for Subjective Question Evaluation

The Sorce Code of "[Towards Human-Like Grading: A Unified LLM-Enhanced Framework for Subjective Question Evaluation](https://arxiv.org/abs/2510.07912)"
 
this paper has been accepted by [ecai2025](https://ecai2025.org/accepted-papers/)

# Structure

```text
├── self-dataset/              # Dataset generation scripts and structure (generation for type1-type4)
│   ├── gen_type1.py/                    # General education (primary & lower grades: Chinese, Math, English)
│   ├── gen_type2.py/                    # Architecture Engineering (National First-/Second-Class Architect exams)
│   ├── gen_type3.py/                    # Computer science (National Computer Rank Examination Level 1/2)
│   ├── gen_type4.py/                    # Humanities-related (public welfare, arts, etc.)  
│   └── data                             # Self-buid dataset
│
├── src_train/                           # Model training module
│   ├── train.py                         # Main training entry; includes lr, batch size, save dir, etc.
│   └── model.py                         # Model components: encoder, scorer, discriminator
│
├── multi_thread_analysis_key_api.py/     # Script for auto-generating keyword information for analysis
│
├── multi_thread_eval_api.py/             # Script for auto-generating evaluation for answer
│
├── multi_thread_key_api.py/              # Script for auto-generating keyword information for answer
│
├── multi_thread_query_api.py/            # Script for generating pseudo-questions
│
├── LICENSE
└── README.md                    # Project documentation (you're reading it!)
```

# Data Generation

```text
python self-dataset/gen_type1.py
python self-dataset/gen_type2.py
...
```
