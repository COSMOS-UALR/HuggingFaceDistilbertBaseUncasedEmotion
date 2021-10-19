# HuggingFace DistilBert Text Classification
A repo for learning how to run the Hugging Face DistilBert model and  calculate text classification. The model used for computing emotions is  bhadresh-savani/distilbert-base-uncased-emotion

# How to run
1. pip install requirements.txt: `pip install -r requirements.txt`
2. run `python huggingFaceDistilBert.py` with arguments found in Command Line Arguments section below.

Example: 
```
python huggingFaceDistilBert.py \
    --input=sample/sample_twitter_data.txt \
    --output=sample/sample_twitter_data_distil_bert.csv 
```

## Command line arguments
`--input`: The input file. Must be a full path to a txt file. For example, `--input=C:/Users/user/Documents/GitHub/HuggingFaceDistilbertBaseUncasedEmotion/sample_twitter_data.txt`.

`--output`: The output file. Must be a full path to a csv file. For example, `--output=C:/Users/user/Documents/GitHub/HuggingFaceDistilbertBaseUncasedEmotion/sample/sample_twitter_data_distil_bert.csv`.




