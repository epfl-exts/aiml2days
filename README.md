<img src="static/Logo-FCUE-2019-cmjn.png" width="180px" align="right">


### AI and ML Essentials - hands-on part

&nbsp;
&nbsp;

This repository contains the material for the hands-on part of the 2 day workshop on **AI and ML Essentials** with the EPFL Extension School.


In this part of the workshop we you will learn about text classification as a supervised learning problem. You will dive into the realm of natural language processing and learn how machines can analyze and learn from text data and text embeddings. 
We will use the [SpamAssassin](https://spamassassin.apache.org/) public email corpus. This dataset contains ~6'000 labeled emails. The dataset has been downloaded for you and is available in the data folder. 

Our aim is to build a spam detector which, given examples of spam emails and examples of regular emails, learns how to flag new emails as spam or non-spam. We will prepare the data, explore it and phrase our problem as a classification task which we solve using a Logistic Regression classifier. 

We will explore and compare various features space and machine learning approaches. The use of spam emails is for demonstration and learning purpose as it is a text-based example that everyone is easily familiar with. Similarly the use of a Logistic Regression model keeps things simple and interate fast through different approaches. This set up allows us to highlight different stages of developing a machine learning application and the decision making processes involved along the way.

## Hands-On Session

To get started with the hands-on session you have the following options. Choose one by clicking on the badges below:


[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/epfl-exts/aiml2days/blob/main/notebooks/data_preparation.ipynb) 
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/epfl-exts/aiml2days/main)
[![Offline](https://img.shields.io/badge/Offline_View-Open-Blue.svg)](https://github.com/epfl-exts/aiml2days/blob/main/static)

- **Colab**: Open the jupyter notebook in **Google Colab** to run the codes interactively on the cloud (recommended for this workshop). Note that you need to have a Google account to run the code in Google Colab.

- **Binder**: You can also interactively run the codes on a server using **Binder**. If you don't have a Google account, you can use this option. 

- **Offline View**: You can choose to take a look at the already executed notebooks in the **Offline View**. Note that with this option you cannot run the codes interactively.


Lastly, should you prefer to run the hands-on session locally on your machine, there are three steps to follow:

1. **Clone or download the content**: Clone this repository from Github to your local machine using the following `git` command in your terminal. Or if you prefer to download the content manually, you can click on the ![](https://placehold.co/60x25/green/white?text=<>+Code) button on the top right of this page and then click on the Download ZIP.
```
git clone https://github.com/epfl-exts/aiml2days.git
```
<br>

1. **Install Miniconda**: Once the content of the repository is on your machine and is extracted, you can install the relevant Python dependencies with `conda`. But before that you need to install `Miniconda` on your system, if you don't have `conda` installed already. Install Miniconda on your system using this [link](https://docs.conda.io/en/latest/miniconda.html).

2. **Installation with conda**: To install the relevant Python dependencies with conda, use the following code in your terminal. This will create a virtual environment called `environment` and install all the necessary packages in it. You can then launch the jupyter notebooks within this environment and run the code interactively.

```
conda env create -f environment.yml
```

**Note**: you need to be in the same folder as the environment.yml file to run this command.