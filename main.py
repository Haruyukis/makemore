from src.modules import basic_neural_net_svc


def main():
    myclass = basic_neural_net_svc.BasicNeuralNetSvc("dev.data.txt")

    # bigram = word_level_bigram_svc.BigramSvc("dev.data.txt")
    # print(bigram.predict_next_word("mandated"))


if __name__ == "__main__":
    main()

# Note to myself
# Words length are between 2 and 19 include
# Example sentences are between 10 and 149 include
