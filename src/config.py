from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ZH_VOCAB_FILE = PROCESSED_DATA_DIR / "vocab_zh.txt"
EN_VOCAB_FILE = PROCESSED_DATA_DIR / "vocab_en.txt"

MODEL_PATH = Path(__file__).parent.parent / "model"

LOGS_DIR = Path(__file__).parent.parent / "logs"

SEQ_LEN = 32
BATCH_SIZE = 128
EMBEDDING_DIM = 128
ENCODER_HIDDEN_DIM = 256
ENCODER_LAYERS = 2

DECODER_HIDDEN_DIM = 2 * ENCODER_HIDDEN_DIM

LEARNING_RATE = 0.001

EPOCHS = 30
