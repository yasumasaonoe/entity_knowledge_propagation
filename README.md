# Can LMs Learn New Entities from Descriptions? Challenges in Propagating Injected Knowledge

This repository contains the data and code for the baseline described in the following paper:

> [**Entity Cloze By Date: What LMs Know About Unseen Entities**](https://aclanthology.org/2023.acl-long.300.pdf) <br/>
> Yasumasa Onoe, Michael J.Q. Zhang, Shankar Padmanabhan, Greg Durrett, Eunsol Choi<br/>
> ACL 2023

## Getting Started

This codebase uses Python 3.7.9. Other versions may work as well.

Dependencies:

```
$ conda create -n ekp -y python=3.7.9
$ conda activate ekp
(ekp) $ pip install -r requirements.txt
```

## Data

- Entity Inferences: `data/entity_inferences`
- ECBD: `data/ecbd`


## Running experiments
From the root dir, run an experiment python file.

Example:
```
(ekp) $ python experiments/gpt_ft.py
```

| Experiment                      | Base Model              | Editing Method | Data              |
|---------------------------------|-------------------------|----------------|-------------------|
| `gpt_ft_ecbd.py`                | GPT2-XL or GPT-Neo 1.3B | Finetuning     | ECBD              |
| `gpt_ft_entity_inferences.py`   | GPT2-XL or GPT-Neo 1.3B | Finetuning     | Entity Inferences |
| `gpt_mend_ecbd.py`              | GPT2-XL                 | MEND           | ECBD              |
| `gpt_mend_entity_inferences.py` | GPT2-XL                 | MEND           | Entity Inferences |
| `t5_ft_ecbd.py`                 | T5-Large                | Finetuning     | ECBD              |
| `t5_ft_entity_inferences.py`    | T5-Large                | Finetuning     | Entity Inferences |
| `t5_mend_ecbd.py`               | T5-Large                | MEND           | ECBD              |
| `t5_mend_entity_inferences.py`  | T5-Large                | MEND           | Entity Inferences |

NOTE: ROME with GPT2-XL will be added soon...


## Citing the paper
```
@inproceedings{onoe-etal-2023-lms,
    title = {{Can LMs Learn New Entities from Descriptions? Challenges in Propagating Injected Knowledge}},
    author = "Onoe, Yasumasa  and
      Zhang, Michael  and
      Padmanabhan, Shankar  and
      Durrett, Greg  and
      Choi, Eunsol",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.300",
    pages = "5469--5485",
}
```

## Contact 

Please contact at `yasumasa@utexas.edu` or `yasumasaonoe@google.com` if you have any questions.