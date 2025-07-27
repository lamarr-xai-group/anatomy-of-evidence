## Anatomy of Evidence: An Investigation Into Explainable ICD Coding 

This repository contains code snippets for analyzing human-annotated evidence and model explanations in the context of explainable medical coding, as presented in the paper [The Anatomy of Evidence: An Investigation Into Explainable ICD Coding](https://aclanthology.org/2025.findings-acl.864/).

### Overview
The code includes the following: 
- Extracting position information of human-annotated evidence spans
- Computing overlap of code descriptions and human-annotated evidence
- Assessing explanation length and model performance
- Computing match measures for evaluating plausibility of model explanations 

### Resources
The analysis utilizes a dataset and model weights from the following repositories: 
- **MDACE dataset** from Cheng et al. (2023) [https://github.com/3mcloud/MDACE](https://github.com/3mcloud/MDACE) - Contains annotated ICD evidence and information how to assemble the dataset
- **Models and evaluation setup** from Edin et al. (2024) [https://github.com/JoakimEdin/explainable-medical-coding](https://github.com/JoakimEdin/explainable-medical-coding) 

### Requirements
The code snippets serve as a reference point. For the analysis of model explanations, it is required that explanations are already extracted and available in a suitable format, such as pickled DataFrame or CSV. 

We are happy to provide further information upon request. 

### Cite as 
```
@inproceedings{beckh-etal-2025-anatomy,
    title = "The Anatomy of Evidence: An Investigation Into Explainable {ICD} Coding",
    author = "Beckh, Katharina  and
      Studeny, Elisa  and
      Gannamaneni, Sujan Sai  and
      Antweiler, Dario  and
      Rueping, Stefan",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2025",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-acl.864/",
    pages = "16840--16851",
    ISBN = "979-8-89176-256-5"
}
```
