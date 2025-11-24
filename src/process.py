import pandas as pd
from sklearn.model_selection import train_test_split

import config
from src.tokenizer import ChineseTokenizer, EnglishTokenizer


def process():
    df = pd.read_csv(config.RAW_DATA_DIR / 'cmn.txt', sep='\t', header=None, usecols=[0, 1], names=['en', 'zh'])

    df = df.dropna()
    df = df[df['en'].str.strip().ne('') & df['zh'].str.strip().ne('')]

    train_df, test_df = train_test_split(df, test_size=0.2)

    ChineseTokenizer.build_vocab(config.ZH_VOCAB_FILE, train_df['zh'].tolist())
    EnglishTokenizer.build_vocab(config.EN_VOCAB_FILE, train_df['en'].tolist())

    zh_tokenizer = ChineseTokenizer.from_vocab(config.ZH_VOCAB_FILE)
    en_tokenizer = EnglishTokenizer.from_vocab(config.EN_VOCAB_FILE)

    # 查看中文句子最大长度
    # print(train_df['zh'].apply(lambda x: len(zh_tokenizer.tokenize(x))).max())

    train_df['zh'] = train_df['zh'].apply(lambda x: zh_tokenizer.encode(x, config.SEQ_LEN, False))
    train_df['en'] = train_df['en'].apply(lambda x: en_tokenizer.encode(x, config.SEQ_LEN, True))

    train_df.to_json(config.PROCESSED_DATA_DIR / 'train_dataset.jsonl', lines=True, orient='records')

    test_df['zh'] = test_df['zh'].apply(lambda x: zh_tokenizer.encode(x, config.SEQ_LEN, False))
    test_df['en'] = test_df['en'].apply(lambda x: en_tokenizer.encode(x, config.SEQ_LEN, True))
    test_df.to_json(config.PROCESSED_DATA_DIR / 'test_dataset.jsonl', lines=True, orient='records')

    print('数据处理完成')


if __name__ == '__main__':
    process()
