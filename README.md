# CRM-GPT


Source code for paper "A GPT-based EHR Modeling System for Unsupervised Novel Disease Detection".

\
**Basic environment:** 

```
conda env create -f environment.yml
```

\
**Code:**

"data/" includes the main code for EHR data pre-processing.

"model/" includes the main code for GPT model building, training and inference.


\
**NOTE:** the BERT (or other LM modules apart from GPT) in `anomaly.py` was used during the development process and partially used as class placeholders in the script, which will not be actually loaded/trained. 

Due to a restrictive data use agreement and HIPAA rules, part of the code for data pre-processing is not provided.

