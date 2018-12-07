import pickle

def load_vocabulary():
    with open('/home/adam/git/nn_dialogue_structure/resources/vocabulary2.pickle', 'rb') as f:
        return pickle.load(f)