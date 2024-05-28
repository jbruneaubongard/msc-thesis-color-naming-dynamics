import pickle
import pandas as pd
import numpy as np
import torch.nn as nn
from torch.optim import Adam
import torch
from torch.distributions import Categorical

PRECISION = 1e-32

cielab = pd.read_csv('../data/munsell_chart.txt', sep='\t')[['L*', 'a*', 'b*']].values
cielab_mixed = np.random.permutation(cielab)
cielab_blank = np.full_like(cielab, 19.)

# Import pU_M
with open("../data/pU_M.pkl", "rb") as f:
    pU_M = pickle.load(f)


class Speaker(nn.Module):
    def __init__(self, vocab_size, hidden_dim=25, layers=1):
        super(Speaker, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        if layers == 1:
            self.model = nn.Sequential(nn.Linear(3, self.hidden_dim),
                                    nn.Sigmoid(),
                                    nn.Linear(self.hidden_dim, self.vocab_size),
                                    nn.Softmax(-1))
        elif layers == 2:
            self.model = nn.Sequential(nn.Linear(3, self.hidden_dim),
                                    nn.Sigmoid(),
                                    nn.Linear(self.hidden_dim, self.hidden_dim),
                                    nn.Sigmoid(),
                                    nn.Linear(self.hidden_dim, self.vocab_size),
                                    nn.Softmax(-1))
            
        elif layers == 3:
            self.model = nn.Sequential(nn.Linear(3, self.hidden_dim),
                                    nn.Sigmoid(),
                                    nn.Linear(self.hidden_dim, self.hidden_dim),
                                    nn.Sigmoid(),
                                    nn.Linear(self.hidden_dim, self.hidden_dim),
                                    nn.Sigmoid(),
                                    nn.Linear(self.hidden_dim, self.vocab_size),
                                    nn.Softmax(-1))
        else:
            self.model = nn.Sequential(nn.Linear(3, self.vocab_size),
                                    nn.Softmax(-1))

    def forward(self, x):
        y = self.model(x)
        dist = Categorical(y)
        action = dist.sample()
        return action, y, dist

class Listener(nn.Module):
    def __init__(self, context_dim, vocab_size, hidden_dim=25):
        super(Listener, self).__init__()
        self.context_dim = context_dim
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.model = nn.Sequential(nn.Embedding(vocab_size, self.hidden_dim),
                                   nn.Sigmoid(),
                                   nn.Linear(self.hidden_dim, self.context_dim),
                                   nn.Softmax(-1))

    def forward(self, x):
        y = self.model(x)
        dist = Categorical(y)
        action = dist.sample()
        return action, y, dist


class Dataset:
    def __init__(self, X, y, cnum, seed=None):
        self.X = X
        self.y = y
        self.cnum = cnum
        self.n_samples = X.size(0)
        self.random_state = np.random.RandomState(seed)

    def full_batch(self):
        return torch.FloatTensor(self.X), torch.LongTensor(self.y)

    def get_cnum_labels(self):
        y = torch.stack(self.y).numpy().tolist()
        return self.cnum, y

    def batch(self, batch_size=32):
        idx = self.random_state.choice(self.n_samples, size=batch_size, replace=True)
        X = torch.stack([self.X[i] for i in idx])
        y = torch.stack([self.y[i] for i in idx])

        return torch.FloatTensor(X), torch.LongTensor(y)


class iteratedLearning:
    def __init__(self, need,
                 vocabulary_size,
                 ib_model,
                 context_dim=330,
                 hidden_dim=25,
                 layers=1,
                 initial_speaker=None,
                 n_episodes=250,
                 batch_size=50,
                 transmission_samples=300,
                 train_steps=1000,
                 learning_rate=0.005):
        '''
        Ini NIL algorithm
        :param need: npy (N,) need distribution to sample colors from
        :param vocabulary_size: (int) maximum number of terms available to the agents
        :param context_dim: (int) size of output space of listener, default is the full munsell chart
        :param initial_speaker: (N, W) npy array. Initial speaker to sample first transmission data from. If None, sample uniform at random
        :param n_episodes: Number of generations
        :param batch_size: Batch size to use at each step
        :param transmission_samples:Size of the transmission dataset
        :param train_steps: Number of training steps in each phase. Total number of samples batch_size * train_steps
        :param learning_rate: Learning rate used in Adam. Optimizer is re-initialized after each phase
        '''
        self.cielab = torch.FloatTensor(cielab)
        self.cielab_mixed = torch.FloatTensor(cielab_mixed)
        self.cielab_blank = torch.FloatTensor(cielab_blank)
        self.need = need
        self.ib_model = ib_model
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.vocabulary_size = vocabulary_size
        self.context_dim = context_dim
        self.n_episodes = n_episodes
        self.batch_size = batch_size
        self.train_steps = train_steps
        self.learning_rate = learning_rate
        self.transmission_samples = transmission_samples
        self.log = []
        self.reset()

        if initial_speaker is None:
            self.initial_speaker = np.ones([330, vocabulary_size]) / vocabulary_size
            self.transmission_phase()
        else:
            self.initial_speaker = initial_speaker
            idx = np.random.choice(330, p=self.need, size=self.transmission_samples, replace=True)
            X = self.cielab[idx]
            y = torch.LongTensor([np.random.choice(vocabulary_size, p=initial_speaker[i]) for i in idx])
            self.dataset = Dataset(X=X, y=y, cnum=idx)



    def run(self):
        '''
        Run NIL for n generations
        :return:
        '''
        # self.log_data()
        convergence = False
        i = 0
        while not convergence and i < self.n_episodes:
            print(f'At generation {i}')
            i += 1
            self.reset()
            self.pre_train_speaker()
            self.pre_train_listener()
            # self.log_data()
            self.interactive_training()
            self.log_data()
            convergence = self.check_convergence()
            self.transmission_phase()
        return self.get_speaker(), i


    def reward(self, x_cielab, y_cielab, c=0.001):
        '''
        Reward used in Kågebäck et al. (2020) based on the similarity measure between colors from Regier et al. 2015
        :param x_cielab: First color represented in 3D cielab
        :param y_cielab: Second color represented in 3D cielab
        :param c: Precision, default 0.001 as in Regier et al. 2015
        :return:
        '''
        diff = (x_cielab - y_cielab)
        dist = diff.norm(2, 1)
        return torch.exp(-c * torch.pow(dist, 2))

    def pre_train_speaker(self):
        optimizer = Adam(self.speaker.parameters(), lr=self.learning_rate)
        CE = nn.CrossEntropyLoss()
        for _ in range(self.train_steps):
            optimizer.zero_grad()
            X, y = self.dataset.batch(batch_size=12)
            _, y_hat, _ = self.speaker(X)
            loss = CE(y_hat, y)
            loss.backward()
            optimizer.step()

    def pre_train_speaker_more_complexity(self, mode):
        self.pre_train_speaker()
        optimizer = Adam(self.speaker.parameters(), lr=self.learning_rate)
        self.log_data()
        convergence = False
        i = 0
        while not convergence and i < self.train_steps:
            i += 1
            optimizer.zero_grad()
            if mode == 'complexity-informativity':
                loss = self.ib_model.accuracy_torch(self.speaker(self.cielab)[1]) - self.ib_model.complexity_torch(self.speaker(self.cielab)[1])
            elif mode == 'complexity':
                loss = - self.ib_model.complexity_torch(self.speaker(self.cielab)[1])
            elif mode == 'informativity-complexity':
                loss = - self.ib_model.accuracy_torch(self.speaker(self.cielab)[1]) + self.ib_model.complexity_torch(self.speaker(self.cielab)[1])
            loss.backward()
            optimizer.step()
            if i!=0 and i%10==0:
                self.log_data()
            convergence = self.check_convergence()

    def check_convergence(self):
        '''
        Test if the speaker and listener have converged. Convergence is defined as no major difference in complexity over 10 generations
        '''
        if len(self.log) < 10:
            return False
        else:
            prev = self.log[-10:]
            prev_complexity = [ self.ib_model.complexity(p) for p in prev]
            prev_accuracy = [ self.ib_model.accuracy(p) for p in prev]
            if np.max(prev_complexity) - np.min(prev_complexity) > 0.05 or np.max(prev_accuracy) - np.min(prev_accuracy) > 0.05:
                return False

        return True
    
    def pre_train_listener(self):
        '''
        Pre-train listener using the pre-trained speaker and REINFORCE
        :return:
        '''
        optimizer = Adam(self.listener.parameters(), lr=self.learning_rate)
        moving_average = 0
        for t in range(self.train_steps):
            optimizer.zero_grad()
            idx = np.random.choice(330, p=self.need, size=self.batch_size, replace=True)
            X = self.cielab[idx]
            word, _, _ = self.speaker(X)
            action, _, dist = self.listener(word.detach())
            guess = self.cielab[action]
            reward = self.reward(X, guess)
            loss = - ((reward - moving_average) * dist.log_prob(action)).mean()
            loss.backward()
            optimizer.step()
            moving_average += 1 / (t + 1) * (reward.mean() - moving_average)

    def interactive_training(self):
        '''
        Play RL reconstruction game using REINFORCE
        :return:
        '''
        optimizer = Adam(list(self.speaker.parameters()) + list(self.listener.parameters()), lr=self.learning_rate)
        moving_average = 0
        for t in range(self.train_steps):
            optimizer.zero_grad()
            idx = np.random.choice(330, p=self.need, size=self.batch_size, replace=True)
            X = self.cielab[idx]
            word, _, dist_speaker = self.speaker(X)
            action, _, dist_listener = self.listener(word)
            guess = self.cielab[action]
            reward = self.reward(X, guess)
            loss = - ((reward - moving_average) * dist_listener.log_prob(action)).mean()\
                   - ((reward - moving_average) * dist_speaker.log_prob(word)).mean()
            loss.backward()
            optimizer.step()
            moving_average += 1 / (t + 1) * (reward.mean() - moving_average)

    def transmission_phase(self):
        '''
        Sample transmission dataset from current speaker and append to next-gen transmission dataset
        :return:
        '''
        idx = np.random.choice(330, p=self.need, size=self.transmission_samples, replace=True)
        X = self.cielab[idx]
        y, _, _ = self.speaker(X)
        self.dataset = Dataset(X=X, y=y, cnum=idx)

    def get_speaker(self):
        '''
        Get speaker over whole Munsell chart
        :return:
        :encoder (330,W) numpy array
        '''
        _, encoder, _ = self.speaker(self.cielab)
        encoder = encoder.detach().numpy()
        return encoder

    def log_data(self):
        self.log.append(self.get_speaker())

    def get_log(self):
        return self.log

    def reset(self):
        '''
        Reset agents for next generation
        :return:
        '''
        self.speaker = Speaker(vocab_size=self.vocabulary_size, hidden_dim=self.hidden_dim, layers=self.layers)
        self.listener = Listener(vocab_size=self.vocabulary_size, context_dim=self.context_dim, hidden_dim=25)

    def run_interaction(self):
        '''
        Run self.n_episodes number of interactions between speaker and listener
        '''
        convergence = False
        i = 0
        # self.log_data()
        while not convergence and i < self.n_episodes:
            i += 1
            print(f'At generation {i}')
            self.interactive_training()
            self.log_data()
            convergence = self.check_convergence()
        return self.get_speaker(), i

    def run_learning(self):
        '''
        Run self.n_episodes of pre-training and transmission
        '''
        convergence = False
        i = 0
        # self.log_data()
        while not convergence and i < self.n_episodes:
            print(f'At generation {i}')
            i += 1
            self.reset()
            self.pre_train_speaker()
            #self.pre_train_listener()
            self.log_data()
            convergence = self.check_convergence()
            self.transmission_phase()
        return self.get_speaker(), i
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('save_path', type=str, help='Where to save the experiment')
    # parser.add_argument('vocab_size', type=int, help='Number of terms available to the agents')
    parser.add_argument('type', default='NIL', type=str, help='Type of experiment to run. NIL, IL, or RL')
    # parser.add_argument('hidden_dim', default=25, type=int, help='Hidden dim')
    # parser.add_argument('layers', default=1, type=int, help='Number of hidden layers')
    args = parser.parse_args()
    return args

