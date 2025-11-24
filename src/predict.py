import torch
from tokenizer import ChineseTokenizer, EnglishTokenizer
import config
from model import TranslationEncoder, TranslationDecoder


def batch_predict(input_tensor, encoder, decoder, en_tokenizer, device):
    encoder.eval()
    decoder.eval()

    batch_size = input_tensor.shape[0]

    with torch.no_grad():
        # input_tensor: [batch_size, seq_len]
        hidden_0 = encoder(input_tensor)
        # hidden_0: [batch_size, hidden_dim]
        hidden_0 = hidden_0.unsqueeze(0)
        # hidden_0: [1, batch_size, hidden_dim]
        decoder_input = torch.full((batch_size, 1), en_tokenizer.sos_index, device=device)
        # decoder_input: [batch_size, 1]

        generate_list = [[] for _ in range(batch_size)]
        is_finished = [False] * batch_size

        for t in range(config.SEQ_LEN):
            decoder_output, hidden_0 = decoder(decoder_input, hidden_0)
            # decoder_output: [batch_size, 1, vocab_size]
            decoder_output = torch.argmax(decoder_output, dim=-1, keepdim=False)
            # decoder_output: [batch_size, 1]
            decoder_output_list = decoder_output.squeeze(1).tolist()
            for i, next_id in enumerate(decoder_output_list):
                if is_finished[i]:
                    continue
                if next_id == en_tokenizer.eos_index:
                    is_finished[i] = True
                    continue
                generate_list[i].append(next_id)

            if all(is_finished):
                break

            decoder_input = decoder_output
    return generate_list


def predict(user_input, zh_tokenizer, en_tokenizer, encoder, decoder, device):
    input_ids = zh_tokenizer.encode(user_input)
    input_tensor = torch.tensor([input_ids]).to(device)
    # input_tensor: [1, seq_len]
    batch_result = batch_predict(input_tensor, encoder, decoder, en_tokenizer, device)
    # batch_result: [[6,7,8],[9,10,11,12]]
    return en_tokenizer.decode(batch_result[0])


def run_predict():
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 分词器
    zh_tokenizer = ChineseTokenizer.from_vocab(config.ZH_VOCAB_FILE)
    en_tokenizer = EnglishTokenizer.from_vocab(config.EN_VOCAB_FILE)

    # 模型
    encoder = TranslationEncoder(zh_tokenizer.vocab_size, zh_tokenizer.pad_index).to(device)
    decoder = TranslationDecoder(en_tokenizer.vocab_size, en_tokenizer.pad_index).to(device)

    encoder.load_state_dict(torch.load(config.MODEL_PATH / "encoder.pt"))
    decoder.load_state_dict(torch.load(config.MODEL_PATH / "decoder.pt"))

    print("请输入中文 q 或 quit 退出")
    while True:
        user_input = input('中文: ')
        if user_input in ['q', 'quit']:
            break
        if user_input.strip() == '':
            continue
        output = predict(user_input, zh_tokenizer, en_tokenizer, encoder, decoder, device)
        print(output)


if __name__ == '__main__':
    run_predict()
