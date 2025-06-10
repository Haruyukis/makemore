from src.modules import data_processing_svc


def main():
    dev_set = data_processing_svc.impl.text_file_reader("dev.data.txt")


if __name__ == "__main__":
    main()

# Note to myself
# Words length are between 2 and 19 include
# Example sentences are between 10 and 149 include
