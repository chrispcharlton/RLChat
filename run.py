import argparse

from _requirements import *
from _config import *

from constants import *
from reinforcement_learning.environment import Env

if __name__ == '__main__':
    """
        Main cmdline invocation.
        
        To train seq2seq model: python run.py train seq2seq
        To train adem model: python run.py train adem
        To train model with rl: python run.py train rl
        Run all cells in Colab to train all. 
        
        Replaces need for train_adem_model.py and train_rl_model.py
    """

    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers(dest='subparser_name')
    train_parser = subparser.add_parser('train')
    train_parser.add_argument(
        'model',
        choices=['seq2seq', 'adem', 'rl', 'discriminator'],
    )

    chat_parser = subparser.add_parser('chat')
    chat_parser.add_argument(
        'model',
        choices=['seq2seq', 'adem', 'rl'],
    )

    args = parser.parse_args()

    if args.subparser_name == 'train':

        if args.model == 'seq2seq':
            from seq2seq.train import train
            train()

        elif args.model == 'adem':
            from ADEM.train import train

            train()

        elif args.model == 'discriminator':
            from Adversarial_Discriminator.train import train
            train()

        elif args.model == "rl":
            from reinforcement_learning.train import train

            num_episodes = 10000
            load_dir = os.path.join(BASE_DIR, SAVE_PATH_SEQ2SEQ)

            policy, env, total_rewards, dqn_losses = train(load_dir=load_dir, num_episodes=num_episodes)
g

    elif args.subparser_name == 'chat':
        """ 
            Temp. 
            Needs Refactor. 
        """
        if args.model == "rl":

            from evaluate.loader2 import load_rl
            from evaluate.inference import get_response_rl


            # import os
            # import torch
            #
            # from seq2seq.models import EncoderRNN, LuongAttnDecoderRNN
            # from seq2seq.vocab import Voc
            #
            # from reinforcement_learning.environment import Env
            # from reinforcement_learning.model import RLGreedySearchDecoder
            from reinforcement_learning import chat
            #
            # from evaluate.inference import get_response_rl
            #
            # # User loader mod and load latest etc
            # latest = '5000_checkpoint.tar'
            # loadFilename = os.path.join(BASE_DIR, SAVE_PATH_RL, latest)
            #
            # checkpoint = torch.load(loadFilename, map_location=device)
            # encoder_sd = checkpoint['en']
            # decoder_sd = checkpoint['de']
            # embedding_sd = checkpoint['embedding']
            # voc = Voc(checkpoint['voc_dict']['name'])
            # voc.__dict__ = checkpoint['voc_dict']
            #
            # print('Building encoder and decoder ...')
            # # Initialize word embeddings
            # embedding = nn.Embedding(voc.num_words, hidden_size)
            # if loadFilename:
            #     embedding.load_state_dict(embedding_sd)
            # # Initialize encoder & decoder models
            # encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
            # decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)
            # if loadFilename:
            #     encoder.load_state_dict(encoder_sd)
            #     decoder.load_state_dict(decoder_sd)
            # # Use appropriate device
            # encoder = encoder.to(device)
            # decoder = decoder.to(device)
            # print('Models built and ready to go!')
            #
            # encoder.eval()
            # decoder.eval()
            #
            # policy = RLGreedySearchDecoder(encoder, decoder, voc)
            # env = Env(voc)

            # evaluate trained model

            latest = '5000_checkpoint.tar'
            loadFilename = os.path.join(BASE_DIR, SAVE_PATH_RL, latest)
            policy, voc = load_rl(loadFilename)
            env = Env(voc)
            env.reset()
            env._state = []

            # print(get_response_rl(policy, env, 'Hi how are you?'))

            chat(policy, env)
