from config.config import config as config
from utils.logger import logger_factory
import pandas as pd
import string
from sklearn.feature_extraction import text
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

# prep the data

# use only titles, disregard body
# remove duplicates
# select N first rows from dataset
# select N most used labels


def prep_data(logger, filename, out_filename, row_limit, out_eval_filename=None, drop_duplicate=True, tag_header="Tags"):
    logger.info(f"RAW datafile: {filename}")

    logger.info(f"Reading datafile, limit: {row_limit}")

    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
    data = pd.read_csv(filename, nrows=row_limit)
    logger.info(f"Read {len(data.index)} rows")

    # Preview the first 5 lines of the loaded data
    # print(data.head())

    # drop null value columns
    data.dropna(inplace=True)

    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.drop.html
    # drop not needed columns
    data = data.drop("Body", axis=1)

    # convert to lower
    if tag_header:
        data["Tags"] = data["Tags"].str.lower()

    data["Title"] = data["Title"].str.lower()

    # Import stopwords with scikit-learn
    stopwords = text.ENGLISH_STOP_WORDS
    # punctuation
    punctuation = string.punctuation
    punctuation = punctuation.replace('#', '')  # let # alone, as to keep C#

    data["Title"] = data['Title'].apply(
        lambda x: ' '.join([word for word in x.split() if word not in stopwords]))
    data["Title"] = data['Title'].apply(
        lambda x: x.translate(str.maketrans('', '', punctuation)))

    # remove duplicate rows
    if drop_duplicate:
        data = data.drop_duplicates(subset="Title", keep="first")

    logger.info(f"After droping duplicates - {len(data.index)} rows")

    if not tag_header:
        data = data.drop("Tags", axis=1)

    # calculate label counts
    # split Tags column by space
    if tag_header:
        data_tags = data["Tags"].str.split(" ", expand=True)
        data['TagsSplit'] = data.Tags.apply(lambda x: x.split(' '))

        logger.info(f"Max label count per data item {data_tags.shape[1]}")
        label_count = data_tags.shape[1]

        data_tags_for_concat = []
        for label_no in range(label_count):
            # data["Tag" + str(label_no)] = data_tags[label_no]
            data_tags_for_concat.append(data_tags[label_no])

        labels = pd.concat(data_tags_for_concat)
        labels_with_counts = labels.value_counts()
        labels_with_counts_keys = labels_with_counts.keys()

        logger.info(f"Unique labels: {len(labels_with_counts)}")

        label_count_to_use = (len(labels_with_counts),
                              config['data']['label_limit'])[len(labels_with_counts) > config['data']['label_limit']]

        for label_no in range(label_count_to_use):
            # data[labels_with_counts_keys[label_no]] = None
            print(f"Label: {label_no} Count: {labels_with_counts[label_no]} Text: {labels_with_counts_keys[label_no]}")

        accepted_label_set = labels_with_counts_keys[0:label_count_to_use]
        print(accepted_label_set)

        # save labels we will use into file - wrong order
        # labels_df = pd.DataFrame(accepted_label_set)
        # labels_df.to_csv(config['data']['label_file_path'], header=False, index=False)

        # remove unwanted labels to keep label count under control
        data["TagsSplit"] = data['TagsSplit'].apply(
            lambda x: list(accepted_label_set & set(x)))

        mlb = MultiLabelBinarizer()
        mlb_df = pd.DataFrame(mlb.fit_transform(data.pop('TagsSplit')), columns=mlb.classes_, index=data.index)

        print("----------------------------")
        print(mlb_df.columns.values)
        labels_alone_df = pd.DataFrame(mlb_df.columns.values)
        labels_alone_df.to_csv(config['data']['label_file_path'], header=False, index=False)

        data = data.join(mlb_df)

        # drop data not needed
        data = data.drop("Tags", axis=1)

    # preview data
    print(data.head())

    # train/eval split
    if out_eval_filename:
        data, eval_data = train_test_split(data, test_size=config['data']['eval_size'])
        logger.info(f"Writing eval data: {out_eval_filename}...")
        eval_data.to_csv(out_eval_filename, header=True, index=False)

    logger.info(f"Writing {out_filename}...")

    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_csv.html
    data.to_csv(out_filename, header=True, index=False)

    logger.info(f"Done!")


def main():
    logger = logger_factory(log_name=config['model']['arch'], log_dir=config['output']['log_dir'])
    logger.info(f"Seed: {config['train']['seed']}")
    prep_data(logger,
              config['data']['raw_train_data_path'], config['data']['train_file_path'],
              config['data']['data_limit_train'], config['data']['validation_file_path'],
              drop_duplicate=True)
#    prep_data(logger,
#              config['data']['raw_test_file_path'], config['data']['test_file_path'], config['data']['data_limit_test'],
#              drop_duplicate=False, tag_header=None)
#    prep_data(logger,
#              config['data']['raw_train_data_path'], config['data']['test_file_path'], config['data']['data_limit_test'],
#              drop_duplicate=True, tag_header=None)


if __name__ == '__main__':
    main()
