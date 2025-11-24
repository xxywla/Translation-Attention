import torch
from nltk.translate.bleu_score import corpus_bleu
from tqdm import tqdm

from dataset import get_dataloader
import config
from tokenizer import ChineseTokenizer, EnglishTokenizer
from model import TranslationEncoder, TranslationDecoder
from predict import batch_predict


def evaluate(dataloader, encoder, decoder, zh_tokenizer, en_tokenizer, device):
    predicts = []
    references = []

    special_tokens = [en_tokenizer.sos_index, en_tokenizer.eos_index, en_tokenizer.pad_index]

    for inputs, targets in tqdm(dataloader, desc='评估'):
        inputs = inputs.to(device)
        batch_predicts = batch_predict(inputs, encoder, decoder, en_tokenizer, device)
        predicts.extend(batch_predicts)

        targets = targets.tolist()
        references.extend([[[index for index in target if index not in special_tokens]] for target in targets])

    return corpus_bleu(references, predicts)


def run_evaluate():
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据集
    dataloader = get_dataloader(is_train=False)

    # 分词器
    zh_tokenizer = ChineseTokenizer.from_vocab(config.ZH_VOCAB_FILE)
    en_tokenizer = EnglishTokenizer.from_vocab(config.EN_VOCAB_FILE)

    # 模型
    encoder = TranslationEncoder(zh_tokenizer.vocab_size, zh_tokenizer.pad_index).to(device)
    decoder = TranslationDecoder(en_tokenizer.vocab_size, en_tokenizer.pad_index).to(device)

    encoder.load_state_dict(torch.load(config.MODEL_PATH / "encoder.pt"))
    decoder.load_state_dict(torch.load(config.MODEL_PATH / "decoder.pt"))

    bleu = evaluate(dataloader, encoder, decoder, zh_tokenizer, en_tokenizer, device)
    print(f'bleu:{bleu}')


if __name__ == '__main__':
    run_evaluate()
