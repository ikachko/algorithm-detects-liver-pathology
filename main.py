import sys
from src.img_reader import IMGReader
from src.model_calculation import *

def main():
    if len(sys.argv) != 2:
        print("Wrong number of arguments")
        return

    img_path = sys.argv[1]

    img_arr = IMGReader.read_image(img_path)

    predictions = calculate_predictions(img_arr)

    pretty_print_labeled_predictions(labeled_predictions(predictions))

    # print_sorted_predictions(predictions)


if __name__ == "__main__":
    main()

