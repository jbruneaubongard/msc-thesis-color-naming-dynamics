import pickle
import tqdm
import argparse as ap
from src.iterated_learning import *
from src.ib_model import *

# Import model
ib_model = load_model()


parser = ap.ArgumentParser()
parser.add_argument(
    '--dataset',
    type=str,
    default='wcs',
    help='Dataset to use. Default: wcs.\
        Chooses the file {dataset}_encoders.pkl',
    )

parser.add_argument(
    '--mode',
    type=str,
    default='NIL',
    help='Method to use. Default: NIL.\
        Available choices: NIL, IL, C',
    choices=['NIL', 'IL', 'C'],
    )


args = parser.parse_args()

# Load encoders
with open(f'../data/{args.dataset}_encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)

# Run the algorithm
for k, initial_encoder in tqdm.tqdm(enumerate(encoders)):
    it = iteratedLearning(need=ib_model.pM.flatten(), 
                            vocabulary_size=np.shape(initial_encoder)[1], 
                            ib_model=ib_model, initial_speaker=initial_encoder, 
                            n_episodes=250)
    if args.mode == 'NIL':
        encoder, t = it.run()
    elif args.mode == 'IL':
        encoder, t = it.run_learning()
    elif args.mode == 'C':
        encoder, t = it.run_interaction()
    
    # complexity = ib_model.complexity(encoder)
    # accuracy = ib_model.accuracy(encoder)"
    # deviation, gnid, beta, _ = ib_model.fit(encoder)

    info = {'Stopping Time': t,
            'Mode': args.mode,
            'n_episodes': it.n_episodes,
            'train_steps': it.train_steps,
            'transmission_samples': it.transmission_samples,
    }

    results = (encoder, info, initial_encoder, it.get_log())

    save_path = f'../results/trajectory_{args.dataset}_{args.mode}_{k}.pkl'

    with open(save_path, 'wb') as f:
        pickle.dump(results, f)
