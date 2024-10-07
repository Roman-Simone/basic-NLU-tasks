<div align="center">
  <h1 style="border-bottom: none;">Basic NLU tasks</h1>
  <img src="https://img.shields.io/badge/python-3670A0?style=flat&logo=python&logoColor=ffdd54" alt="Python"/>
  <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/nltk-85C1E9?style=flat&logoColor=white" alt="NLTK"/>
</div>

## Table of Contents

- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Running the Project](#running-the-project)
- [Contact](#contact)

## Project Overview
This project introduces the main tasks in the field of Natural Language Understanding (NLU). The following tasks are covered:

1. **Building a Neural Language Model (LM)**:
   - Using **LSTM** with regularization techniques to improve performance.
   
2. **Intent Classification and Slot Filling**:
   - First with LSTM,
   - Then using **BERT** for enhanced results.

3. **Aspect-Based Sentiment Analysis (ABSA)**:
   - Focused on term extraction using BERT.

Each folder contains a detailed report outlining the corresponding tasks.

## Project structure
```
basic_nlu_tasks
├── LM
│   ├── part_1
│   │   ├── dataset
│   │   ├── extra
│   │   │   ├── details_models
│   │   │   ├── test_model.py
│   │   ├── functions.py
│   │   ├── main.py
│   │   ├── model.py
│   │   ├── README.md
│   │   └── utils.py
│   └── part_2
├── NLU
│   ├── part_1
│   └── part_2
├── nlu_env.yaml
├── README.md
├── requirements.txt
└── SA
    └── part_1
```
- LM: Language models
- NLU: Slot filling and intent recognition
- SA: Aspect extraction for Sentiment Analyisis

## Installation

In order to run the project you'll need to clone it and install the requirements. We suggest you to create a virtual environment 
- Clone it

    ```BASH
    git clone https://github.com/Roman-Simone/basic_nlu_tasks

    ```
- Create the env, in this case with conda but venv could be also used:

    ```bash
    conda env create -f nlu_env.yaml -n nlu24
    conda activate nlu24
    ``` 

## Running the project
In order to run the examples enter in the folder of the task and run the `main.py` file

# Contact
For any inquiries, feel free to contact:

- Simone Roman - [simone.roman@studenti.unitn.it](mailto:simone.roman@studenti.unitn.it)

<br>

<div>
    <a href="https://www.unitn.it/">
        <img src="https://ing-gest.disi.unitn.it/wp-content/uploads/2022/11/marchio_disi_bianco_vert_eng-1024x295.png" width="400px">
    </a>
</div>
