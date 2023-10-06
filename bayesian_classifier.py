# Write the code for bayesian classifier algo here

def initialize_training_data():
    pass


def train_algorithm():
    estimate_y_values()
    estimate_x_values()

    # Use MLE to train algo
    mle()
    classify_xnew()

    # Alternatively, use MAP

def main():
    initialize_training_data()
    train_algorithm()


if __name__ == "__main__":
    main()