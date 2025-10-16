from utils import create_input_files
import time
import argparse

if __name__ == '__main__':

    print('create_input_files START at: ', time.strftime("%m-%d  %H : %M : %S", time.localtime(time.time())))
    parser = argparse.ArgumentParser(description = 'Create Input Files')
    parser.add_argument('--dataset', default = "LEVIR_CC")
    parser.add_argument('--karpathy_json_path', default = "./Levir_CC_dataset/LevirCCcaptions.json")
    parser.add_argument('--image_folder', default = "./Levir_CC_dataset/images")
    parser.add_argument('--captions_per_image', type = int, default = 5)
    parser.add_argument('--min_word_freq', type = int, default = 5)
    parser.add_argument('--output_folder', default = './data')
    parser.add_argument('--max_len', type = int, default = 50)
    args = parser.parse_args()
    

    create_input_files(dataset=args.dataset,
                       karpathy_json_path=args.karpathy_json_path,
                       image_folder=args.image_folder,
                       captions_per_image=args.captions_per_image,
                       min_word_freq=args.min_word_freq,
                       output_folder=args.output_folder,
                       max_len=args.output_folder)

    print('create_input_files END at: ', time.strftime("%m-%d  %H : %M : %S", time.localtime(time.time())))