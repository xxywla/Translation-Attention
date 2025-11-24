import time
from itertools import chain

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.dataset import get_dataloader
from src.model import TranslationEncoder, TranslationDecoder
from src.tokenizer import ChineseTokenizer, EnglishTokenizer
import config


def train_one_epoch(dataloader, encoder, decoder, en_tokenizer, loss_function, optimizer, device):
    encoder.train()
    decoder.train()

    epoch_total_loss = 0

    for inputs, targets in tqdm(dataloader):
        optimizer.zero_grad()

        inputs, targets = inputs.to(device), targets.to(device)
        # inputs: [batch_size, seq_len]
        context_vector = encoder(inputs)
        # hidden_0: [batch_size, hidden_dim*2]
        hidden_0 = context_vector.unsqueeze(0)
        # hidden_0: [1, batch_size, hidden_dim*2]
        decoder_input = targets[:, 0:1]
        # decoder_input: [batch, 1]

        generate_list = []

        for t in range(1, targets.shape[1]):
            decoder_output, hidden_0 = decoder(decoder_input, hidden_0)
            # decoder_output: [batch_size, 1, vocab_size]
            decoder_input = targets[:, t:t + 1]
            # decoder_input: [batch_size, 1]
            generate_list.append(decoder_output)

        predict_result = torch.cat(generate_list, dim=1)
        # predict_result: [batch_size, seq_len, vocab_size]

        loss = loss_function(predict_result.reshape(-1, predict_result.shape[-1]), targets[:, 1:].reshape(-1))
        loss.backward()
        optimizer.step()

        epoch_total_loss += loss.item()

    return epoch_total_loss / len(dataloader)


def run_train():
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 数据集
    dataloader = get_dataloader()
    # 分词器
    zh_tokenizer = ChineseTokenizer.from_vocab(config.ZH_VOCAB_FILE)
    en_tokenizer = EnglishTokenizer.from_vocab(config.EN_VOCAB_FILE)

    # 模型
    encoder = TranslationEncoder(zh_tokenizer.vocab_size, zh_tokenizer.pad_index).to(device)
    decoder = TranslationDecoder(en_tokenizer.vocab_size, en_tokenizer.pad_index).to(device)

    # 损失函数
    loss_function = nn.CrossEntropyLoss(ignore_index=en_tokenizer.pad_index)

    # 优化器
    optimizer = torch.optim.Adam(chain(encoder.parameters(), decoder.parameters()), lr=config.LEARNING_RATE)

    # tensorboard
    writer = SummaryWriter(log_dir=config.LOGS_DIR / time.strftime('%Y-%m-%d_%H-%M-%S'))

    best_loss = float("inf")

    for epoch in range(1, config.EPOCHS + 1):
        print(f"==== Epoch {epoch} =====")
        avg_loss = train_one_epoch(dataloader, encoder, decoder, en_tokenizer, loss_function, optimizer, device)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(encoder.state_dict(), config.MODEL_PATH / "encoder.pt")
            torch.save(decoder.state_dict(), config.MODEL_PATH / "decoder.pt")
            print("模型保存成功")
        else:
            print("无需保存")

        writer.add_scalar("loss", avg_loss, epoch)
        print(f"Epoch {epoch}/{config.EPOCHS}, Loss: {avg_loss}")


if __name__ == '__main__':
    run_train()
