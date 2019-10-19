import argparse
from constants import *



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
        choices=['seq2seq', 'adem', 'rl'],
    )

    args = parser.parse_args()

    if args.subparser_name == 'train':

        if args.model == 'seq2seq':
            from seq2seq.train import train
            train()

        elif args.model == 'adem':
            from ADEM import train
            train()

        elif args.model == "rl":
            from reinforcement_learning import train, chat

            policy, env, total_rewards, dqn_losses = train(num_episodes=50)

            # evaluate trained model
            chat(policy, env)
