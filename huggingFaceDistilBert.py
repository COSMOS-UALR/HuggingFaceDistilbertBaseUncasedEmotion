import pandas as pd
import os
import argparse
import time
from transformers import pipeline

def main(args):
    # validate input file
    if not os.path.isfile(args.input):
        raise ValueError(f"Input file does not exist. No file located at: {args.input}")
    if not args.input.endswith('.txt'):
        raise ValueError("Input file must be a .txt file")

    # validate output file
    if not args.output.endswith('.csv'):
        raise ValueError("Output file must be a .csv file")

    # read the textFile
    df = pd.read_csv(args.input,names=['text'], sep="\n", header=None)

    # define the distilbert text-classifier
    classifier = pipeline("text-classification",model='bhadresh-savani/distilbert-base-uncased-emotion', return_all_scores=True)

    # method to classify the text and calculate elapsed time in seconds
    def get_distil_bert_textClassification(txt):
        t = time.process_time()
        result = classifier(txt)
        elapsed_time = time.process_time() - t
        return pd.Series([result, elapsed_time])

    # apply get_distil_bert_textClassification() function to column 'text' and assign it to classification_score column
    df[['classification_score', 'elapsed_time']]= df['text'].apply(get_distil_bert_textClassification)

    # flatten the classification_score
    df['classification_score_flatten'] = df['classification_score'].apply(lambda x: x[0])
    list = pd.DataFrame.from_records(df['classification_score_flatten'],
                                 columns=["sadness", "joy", "love", "anger", "fear", "surprise"])

    df["sadness"]  = list['sadness'].apply(lambda x: x['score'])
    df["joy"]      = list['joy'].apply(lambda x: x['score'])
    df["love"]     = list['love'].apply(lambda x: x['score'])
    df["anger"]    = list['anger'].apply(lambda x: x['score'])
    df["fear"]     = list['fear'].apply(lambda x: x['score'])
    df["surprise"] = list['surprise'].apply(lambda x: x['score'])

    # Save the dataframe to CSV file.
    df.drop(columns=['classification_score', 'classification_score_flatten'])\
        .to_csv(args.output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="input file", required=True)
    parser.add_argument("--output", help="output file", required=True)
    args = parser.parse_args()
    main(args)
