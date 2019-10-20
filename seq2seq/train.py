from _requirements import *
from seq2seq.models import EncoderRNN, LuongAttnDecoderRNN
from seq2seq.trainingMethods import trainEpochs
from seq2seq.vocab import Voc
from data.amazon.dataset import AlexaDataset
from torch.utils.data import DataLoader
from _config import *
from constants import *


def train():

    save_dir = os.path.join("data", "save")
    data = AlexaDataset()
    voc = Voc.from_dataset(data)

    # Set checkpoint to load from; set to None if starting from scratch
    loadFilename = None
    #loadFilename = os.path.join(save_dir, model_name, corpus_name, '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers,
    #                               hidden_size), '{}_checkpoint.tar'.format(checkpoint_iter))

    # Load model if a loadFilename is provided
    if loadFilename:
        # If loading on same machine the model was trained on
        checkpoint = torch.load(loadFilename)
        # If loading a model trained on GPU to CPU
        #checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
        encoder_sd = checkpoint['en']
        decoder_sd = checkpoint['de']
        encoder_optimizer_sd = checkpoint['en_opt']
        decoder_optimizer_sd = checkpoint['de_opt']
        embedding_sd = checkpoint['embedding']
        voc.__dict__ = checkpoint['voc_dict']
    else:
        checkpoint = None

    print('Building encoder and decoder ...')
    # Initialize word embeddings
    embedding = nn.Embedding(voc.num_words, hidden_size)
    if loadFilename:
        embedding.load_state_dict(embedding_sd)
    # Initialize encoder & decoder models
    encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
    decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)
    if loadFilename:
        encoder.load_state_dict(encoder_sd)
        decoder.load_state_dict(decoder_sd)

    # Use appropriate device
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    print('Models built and ready to go!')

    # Ensure dropout layers are in train mode
    encoder.train()
    decoder.train()

    # Initialize optimizers
    print('Building optimizers ...')
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
    if loadFilename:
        encoder_optimizer.load_state_dict(encoder_optimizer_sd)
        decoder_optimizer.load_state_dict(decoder_optimizer_sd)

    # If you have cuda, configure cuda to call
    for state in encoder_optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()

    for state in decoder_optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()

    train_data = AlexaDataset('train.json', rare_word_threshold=3)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    corpus_name = 'Alexa'

    # Run training iterations
    print("Starting Training!")
    trainEpochs(model_name, voc, n_epochs, train_loader, encoder, decoder, encoder_optimizer, decoder_optimizer, embedding,
               encoder_n_layers, decoder_n_layers, save_dir, print_every, clip, corpus_name, loadFilename, checkpoint, hidden_size)


if __name__ == '__main__':
    train()
