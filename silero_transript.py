import warnings

warnings.filterwarnings("ignore", category=UserWarning)
import torch


# 1. Load the model and utility functions
# Languages available: 'en', 'de', 'es', 'ru'
device = torch.device("cpu")
model, decoder, utils = torch.hub.load(
    repo_or_dir="snakers4/silero-models",
    model="silero_stt",
    language="en",
    device=device,
)

(read_batch, split_into_batches, read_audio, prepare_model_input) = utils


# Silero STT works best with 16kHz mono audio

test_files = [r"E:\ML\pauses\load_dir\new.wav"]
batches = split_into_batches(test_files, batch_size=10)
input = prepare_model_input(read_batch(batches[0]), device=device)

# 3. Run Inference

output = model(input)

# 4. Decode the output into readable text

for example in output:
    print(decoder(example.cpu()))

# print(output)
