from input import get_batch_generators
from model import create_model
from vocabulary import load_vocabulary

def main():
    vocabulary = load_vocabulary()

    train, val = get_batch_generators()
    
    model = create_model(vocabulary)

    #
    items_in_sequence = 1024
    batch_size = 32
    x = items_in_sequence/batch_size

    model.fit_generator(generator=train, 
                        validation_data=val,
                        validation_steps=x,
                        steps_per_epoch=x,
                        epochs=3,
                        verbose=2)


if __name__ == '__main__':
    main()