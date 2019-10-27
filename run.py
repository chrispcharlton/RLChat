import argparse
#import colorama
from typing import Text

from _requirements import *
from _config import *

from constants import *
from data.amazon.dataset import AlexaDataset, standardise_sentence
from reinforcement_learning.environment import Env


from seq2seq.models import EncoderRNN, LuongAttnDecoderRNN
from seq2seq.vocab import Voc

from reinforcement_learning.environment import Env
from reinforcement_learning.model import RLGreedySearchDecoder


def load(file_path, dataset):
    checkpoint = torch.load(file_path, map_location=device)
    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']
    embedding_sd = checkpoint['embedding']
    voc = Voc(checkpoint['voc_dict']['name'])
    voc.__dict__ = checkpoint['voc_dict']

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
    encoder.eval()
    decoder.eval()

    policy = RLGreedySearchDecoder(encoder, decoder, voc)
    env = Env(voc, dataset)
    return policy, env


def get_response(policy, env, utterance) -> Text:
    """

    :param policy:
    :param env:
    :param utterance: Text
    :return:
    """
    try:
        message_tensor = env.sentence2tensor(standardise_sentence(utterance))
        env.update_state(message_tensor)
        response, tensor = policy.response(env.state)
        env.update_state(tensor)
    except KeyError:
        response = "Error: Encountered unknown word."

    return response


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

    results_parser = subparser.add_parser('results')
    results_parser.add_argument(
        'model',
        choices=['all'],
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

    elif args.subparser_name == 'chat':
        """ 
            Temp. 
            Needs Refactor. 
        """
        if args.model == "rl":
            pass

    elif args.subparser_name == 'results':

        if args.model == 'all':

            utterances = [
                'Hi, how are you?',
                'Nice to see you again.',
                "Shocking weather today aint it?",
                'Hello you',
                'Did you have a pleasant weekend?',
                'I really like green eggs and ham...',
                "We're friends",
                'There are 89 horse races on today, I am hoping to watch at least 6 of them',
                "You're a nice guy, maybe we could go out for a game of bowls...",
                'I am always on the computer',

                "Do you watch or keep up on much basketball? It's definitely a team sport I didn't play much of",
                "good morning! do you like football?",
                "howdy there. do you like radio? ",
                "Good afternoon, hope it's going well for you. Are you interested in politics? ",
                "Hello! How are you tonight?",
                "I have a friend who is considering engineering, and I should tell her that she'll make more in her lifetime than the average NFL or MLB player",
                "Hi, how are you? Do you know much about the Bible?",
                "Hello, do you watch the NFL?",
                "Have you heard about the pictures of Odell Beckham doing cocaine?",
                "Are you more of a baseball or football fan?",
            ]

            num_cols = 5
            num_rows = 20
            rows = [[None for i in range(num_cols)] for j in range(num_rows)]

            dataset = AlexaDataset()
            saves = [
                '1000_checkpoint.tar',
                'Discriminator_1000_checkpoint.tar',
                '1000_checkpoint_mixed_reward.tar',
            ]

            for i, save in enumerate(saves):
                loadFilename = os.path.join(BASE_DIR, SAVE_PATH_RL, save)
                policy, env = load(loadFilename, dataset)

                for j, u in enumerate(utterances):
                    env.reset()
                    env._state = []
                    r = get_response(policy, env, u)

                    rows[j][0] = u
                    rows[j][i +1] = r
                    #print(u, colorama.Fore.MAGENTA + r + colorama.Fore.RESET)

            with open(os.path.join(BASE_DIR, RESULTS, 'latest_results.csv'), mode='w') as f:
                res_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                for line in rows:
                    res_writer.writerow(line)


