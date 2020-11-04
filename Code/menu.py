import os
from utils import bcolors
from utils import get_relations_by_entityID, extracts_prediction_samples, all_label_property, print_stats_validation_get



startWorkPath = os.path.abspath(os.getcwd())
basic_folders = ['./data/relations', './data/labels', './data/splits/known',
                    './data/splits/zero_shot', './train/known', './train/zero_shot',
                    './data/predictions', './data/wiki', './out/features/plain/known', './out/features/trainonknown/known',
                    './out/features/trainonknown/zero_shot', './out/logs', './out/predictions/known', './out/predictions/zero_shot',
                    './base_model', './data/model']

numberOfGPUS = 1

def create_all_folders():
    currentPath = os.path.abspath(os.getcwd())
    if currentPath.split('/')[-1] == 'kbCompletion-tmp':
        for folder_path in basic_folders:
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
        print(f'{bcolors.OKGREEN}=============== FINISH CREATING FOLDERS ==============={bcolors.ENDC}')
        input('[ENTER] to continue')
    else:
        print(f'{bcolors.FAIL}========== FAILED: In the Wrong folder =========={bcolors.ENDC}')

"""
    Function to download all the Wikipedia articles
"""
def download_wikipedia_articles():
    command = 'python src/download_wikipedia.py ./data/relations ./data/wiki'
    print('COMMAND: ', command)
    os.system(command)

def download_wikipedia_articles_predict():
    command = 'python src/download_wikipedia.py ./data/splits/zero_shot ./data/wiki'
    print('COMMAND: ', command)
    os.system(command)

"""
    Function to split dataset
"""
def split_data(ratio=0.9):
    command = 'python src/split_data.py --input_path="./data/relations" --wiki_path="./data/wiki" --save_path="./data/splits" --ratio=' + str(ratio)
    print('COMMAND: ' + command)
    os.system(command)

"""
    Function to generate training dataset for BERT Model
"""
def generate_training_data():
    print(f'{bcolors.OKGREEN} ========== START GENERATING POSITIVE SAMPLES =========={bcolors.ENDC}')
    command = "python src/generate_training_data.py --input_path=./data/splits/known --wiki_path=./data/wiki --save_path=./data/train/known --labels_path=./data/labels --data_type='train'"
    os.system(command)
    print('COMMAND: ' + command)
    print(f'{bcolors.OKGREEN} ========== FINISH GENERATING POSITIVE SAMPLES =========={bcolors.ENDC}')
    genarate_negative_data()

def genarate_negative_data():
    print(f'{bcolors.OKGREEN} ========== START GENERATING NEGATIVE SAMPLES =========={bcolors.ENDC}')
    command = "python generate_negative_sample.py"
    os.system(command)
    print('COMMAND:' + command)
    print(f'{bcolors.OKGREEN} ========== FINISH GENERATING NEGATIVE SAMPLES =========={bcolors.ENDC}')

def bert_training():
    """
    CUDA_VISIBLE_DEVICES=0,1,2 python src/train_bert/run_squad.py  --vocab_file=../bert/uncased_L-12_H-768_A-12/vocab.txt --bert_config_file=../bert/uncased_L-12_H-768_A-12/bert_config.json --init_checkpoint=../bert/train/model.ckpt-21719 --do_train=True --do_predict=False --train_file=./data/train/known/known_relations_train.json --train_batch_size=3 --learning_rate=3e-5 --num_train_epochs=2.0 --max_seq_length=100 --doc_stride=128 --output_dir=./data/model --do_lower_case=True --version_2_with_negative=True --null_score_diff_threshold=-1.8903636932373047
    """
    command = 'CUDA_VISIBLE_DEVICES={} python src/train_bert/run_squad.py  --vocab_file=./base_model/vocab.txt --bert_config_file=./base_model/bert_config.json --init_checkpoint=./base_model/model.ckpt-21719 --do_train=True --do_predict=False --train_file=./data/train/known/known_relations_train.json --train_batch_size=3 --learning_rate=3e-5 --num_train_epochs=2.0 --max_seq_length=100 --doc_stride=128 --output_dir=./data/model --do_lower_case=True --version_2_with_negative=True --null_score_diff_threshold=-1.8903636932373047'

    numberOfGPUS = int(input('Enter the number of GPU in you machine: '))
    if(numberOfGPUS == 1):
        print('COMMAND:' + command.format('0'))
        os.system(command.format('0'))
    elif numberOfGPUS == 2:
        print('COMMAND:' + command.format('0,1'))
        os.system(command.format('0,1'))
    elif numberOfGPUS == 3:
        print('COMMAND:' + command.format('0,1,2'))
        os.system(command.format('0,1,2'))
    elif numberOfGPUS == 4:
        print('COMMAND:' + command.format('0,1,2,3'))
        os.system(command.format('0,1,2,3'))
    else:
        print('WRONG INPUT, should enter between 1~4')

def relation_extration():
    command = 'CUDA_VISIBLE_DEVICES={} python src/relation_extraction.py --feat_path=out/features/trainonknown/known' + \
              ' --split=known --wiki_data=./data/wiki --vocab_file=./base_model/vocab.txt --bert_config_file=./base_model/bert_config.json --init_checkpoint=./data/model --output_dir=/tmp/tmp1 --do_predict=True' + \
              ' --do_train=False --predict_file=./ --k_sentences=20 --predict_batch_size=32 --num-kw-queries=5 --out_name=kw_sent --version_2_with_negative=True' + \
              ' --null_score_diff_threshold=-1.8903636932373047'
    if (numberOfGPUS == 1):
        print('COMMAND:' + command.format('0'))
        os.system(command.format('0'))
    elif numberOfGPUS == 2:
        print('COMMAND:' + command.format('0,1'))
        os.system(command.format('0,1'))
    elif numberOfGPUS == 3:
        print('COMMAND:' + command.format('0,1,2'))
        os.system(command.format('0,1,2'))
    elif numberOfGPUS == 4:
        print('COMMAND:' + command.format('0,1,2,3'))
        os.system(command.format('0,1,2,3'))
    else:
        print('WRONG INPUT, should enter between 1~4')

def train_ranker():
    os.chdir(startWorkPath + '/src')
    command = "python train_ranker.py --querytype='kw_sent' --experiment='trainonknown' --val_type='known'"
    print('COMMAND:' + command)
    os.system(command)
    os.chdir(startWorkPath)

def all_in_one():
    download_wikipedia_articles()
    split_data()
    generate_training_data()
    bert_training()
    relation_extration()
    train_ranker()

def generate_predict_samples(entityId, relations_arr, amountOfSamples):
    for relation in relations_arr:
        # if relation[0] == 'http://www.wikidata.org/prop/direct/P31':
        #     continue
        extracts_prediction_samples(entityId=entityId, property=relation[0], amount=amountOfSamples)
        all_label_property(property=relation[0])

def get_labels(property):
    all_label_property(property=property)

def validate_neg():
    command = 'CUDA_VISIBLE_DEVICES={} python src/relation_extraction.py --feat_path=out/features/trainonknown/zero_shot --split=neg --wiki_data=./data/wiki --vocab_file=/home/guo/bert/vocab.txt --bert_config_file=/home/guo/bert/bert_config.json --init_checkpoint=/home/guo/trainonall --output_dir=/tmp/tmp1 --do_predict=True --do_train=False --predict_file=./ --k_sentences=20 --predict_batch_size=32 --num-kw-queries=5 --out_name=kw_sent --version_2_with_negative=True --null_score_diff_threshold=-1.8903636932373047'
    numberOfGPUS = int(input('Enter the number of GPU in you machine: '))
    if (numberOfGPUS == 1):
        print('COMMAND:' + command.format('0'))
        os.system(command.format('0'))
    elif numberOfGPUS == 2:
        print('COMMAND:' + command.format('0,1'))
        os.system(command.format('0,1'))
    elif numberOfGPUS == 3:
        print('COMMAND:' + command.format('0,1,2'))
        os.system(command.format('0,1,2'))
    elif numberOfGPUS == 4:
        print('COMMAND:' + command.format('0,1,2,3'))
        os.system(command.format('0,1,2,3'))
    print_stats_validation_get()

def train_pipeline():
    relations = [f.split('.json')[0] for f in os.listdir('./data/relations') if f.endswith('.json') and f.startswith('P')]
    print(f'{bcolors.WARNING}>>> Got {len(relations)} relations in the "./data/relations folder" <<<{bcolors.ENDC}')
    while(True):
        print_subtrain_menu()
        option = input()
        if (option == '1'):
            download_wikipedia_articles()
        elif(option == '2'):
            ratio = float(input('Enter ratio to split train/test: '))
            if isinstance(ratio, float):
                if ratio > 0 and ratio <= 1.0:
                    split_data(ratio)
                else:
                    split_data()
            elif isinstance(ratio, str):
                print(f'{bcolors.FAIL}Wrong Input, Use default ratio => 0.9{bcolors.ENDC}')
                split_data()
        elif(option == '3'):
            generate_training_data()
        elif(option == '4'):
            bert_training()
        elif(option == '5'):
            relation_extration()
        elif(option == '6'):
            train_ranker()
        elif(option == '7'):
            all_in_one()
        elif(option == '8'):
            validate_neg()
        elif (option == '0'):
            break;
        else:
            print('Wrong option entered !')

def relation_extration_predict():
    command = 'CUDA_VISIBLE_DEVICES={} python src/relation_extraction.py --feat_path=out/features/trainonknown/zero_shot' + \
              ' --split=zero_shot --wiki_data=./data/wiki --vocab_file=/home/guo/bert/vocab.txt' + \
              ' --bert_config_file=/home/guo/bert/bert_config.json --init_checkpoint=/home/guo/trainonall' + \
              ' --output_dir=/tmp/tmp1 --do_predict=True --do_train=False --predict_file=./ --k_sentences=20' + \
              ' --predict_batch_size=32 --num-kw-queries=5 --out_name=kw_sent --version_2_with_negative=True --null_score_diff_threshold=-1.8903636932373047'
    numberOfGPUS = int(input('Enter the number of GPU in you machine: '))
    if (numberOfGPUS == 1):
        print('COMMAND:' + command.format('0'))
        os.system(command.format('0'))
    elif numberOfGPUS == 2:
        print('COMMAND:' + command.format('0,1'))
        os.system(command.format('0,1'))
    elif numberOfGPUS == 3:
        print('COMMAND:' + command.format('0,1,2'))
        os.system(command.format('0,1,2'))
    elif numberOfGPUS == 4:
        print('COMMAND:' + command.format('0,1,2,3'))
        os.system(command.format('0,1,2,3'))
    else:
        print('WRONG INPUT, should enter between 1~3')

def train_ranker_predict():
    os.chdir(startWorkPath + '/src')
    command = "python train_ranker.py --querytype='kw_sent' --experiment='trainonknown' --val_type='zero_shot'"
    print('COMMAND:' + command)
    os.system(command)
    os.chdir(startWorkPath)

def predict_newfacts():
    relations = [f.split('_test.json')[0] for f in os.listdir('./data/splits/zero_shot') if
                 f.endswith('_test.json') and f.startswith('P')]
    while (True):
        print_predict_menu()
        option = input()
        if (option == '1'):
            print(f'{bcolors.WARNING}>>> Got {len(relations)} relations in the "./data/splits/zero_shot folder" <<<{bcolors.ENDC}')
            relation_extration_predict()
            train_ranker_predict()
        elif (option == '2'):
            entityID = input('Enter the Id of entity[e.g. Q11032]: ')
            num_k = int(input('Enter the number k => top-k relations: '))
            top_k_relations = get_relations_by_entityID(entityID=entityID, k=num_k)
            print_top_k_relations(entityID, top_k_relations)
        elif (option == '3'):
            print('Not finished yet ~')
            pass
        elif (option == '0'):
            break
        else:
            print(f'{bcolors.FAIL}Wrong input, enter option between 1~3{bcolors.ENDC}')

"""
          =============== ALL MENU PRINT FUNCTIONS ==============
"""
def print_menu():
    print(f'{bcolors.OKGREEN}====================== Main Menu ======================{bcolors.ENDC}')
    print(f'{bcolors.UNDERLINE} -- 1. Run the pipeline[TRAIN]{bcolors.ENDC}')
    print(f'{bcolors.UNDERLINE} -- 2. Predict facts[PREDICT]{bcolors.ENDC}')
    print(f'{bcolors.UNDERLINE} -- 3. Create initial folders [FIRST TIME]{bcolors.ENDC}')
    print(f'{bcolors.UNDERLINE} -- 4. INFO [README] {bcolors.ENDC}')
    print(f'{bcolors.UNDERLINE} -- 0. Exit {bcolors.ENDC}')
    print(f'{bcolors.OKGREEN}========================= End ========================={bcolors.ENDC}')

    print(f'{bcolors.BOLD}CURRENT FOLDER: {os.path.abspath(os.getcwd())} {bcolors.ENDC}', end='\n')
    if(os.path.abspath(os.getcwd()).split('/')[-1] != 'kbCompletion-tmp'):
        print(f'{bcolors.FAIL}WRONG FOLDER => Re-run the script in the right folder plz{bcolors.ENDC}')
        return -1
    print(f'{bcolors.OKBLUE}Enter your option:{bcolors.ENDC}', end='')
    return 0

def print_subtrain_menu():
    print(f'{bcolors.OKGREEN}====================== Train Menu ======================{bcolors.ENDC}')
    print(f'{bcolors.UNDERLINE} -- 1. Download wikipedia articles {bcolors.ENDC}')
    print(f'{bcolors.UNDERLINE} -- 2. Split data [default ratio=0.9]{bcolors.ENDC}')
    print(f'{bcolors.UNDERLINE} -- 3. Generate training data {bcolors.ENDC}')
    print(f'{bcolors.UNDERLINE} -- 4. Run relations over BERT {bcolors.ENDC}')
    print(f'{bcolors.UNDERLINE} -- 5. Relation Extraction [Take more time] {bcolors.ENDC}')
    print(f'{bcolors.UNDERLINE} -- 6. Pass to RankerNet {bcolors.ENDC}')
    print(f'{bcolors.UNDERLINE} -- 7. Run option 1~6 [ALL IN ONE]{bcolors.ENDC}')
    print(f'{bcolors.UNDERLINE} -- 8. Validate negative part {bcolors.ENDC}')
    print(f'{bcolors.UNDERLINE} -- 0. Return {bcolors.ENDC}')
    print(f'{bcolors.OKGREEN}====================== End ======================={bcolors.ENDC}')
    print(f'{bcolors.BOLD}CURRENT FOLDER: {os.path.abspath(os.getcwd())} {bcolors.ENDC}', end='\n')
    print(f'{bcolors.OKBLUE}Enter your option:{bcolors.ENDC}', end='')

def print_predict_menu():
    print(f'{bcolors.OKGREEN}====================== PREDICT MENU ======================{bcolors.ENDC}')
    print(f'{bcolors.UNDERLINE} -- 1. Find by all the relations in the zero_shot folder{bcolors.ENDC}')
    print(f'{bcolors.UNDERLINE} -- 2. Find by category[e.g. Q11032]{bcolors.ENDC}')
    print(f'{bcolors.UNDERLINE} -- 3. Find by article link{bcolors.ENDC}')
    print(f'{bcolors.UNDERLINE} -- 0. Return {bcolors.ENDC}')
    print(f'{bcolors.OKGREEN}========================= End ========================={bcolors.ENDC}')

    print(f'{bcolors.BOLD}CURRENT FOLDER: {os.path.abspath(os.getcwd())} {bcolors.ENDC}', end='\n')
    if (os.path.abspath(os.getcwd()).split('/')[-1] != 'kbCompletion-tmp'):
        print(f'{bcolors.FAIL}WRONG FOLDER => Re-run the script in the right folder plz{bcolors.ENDC}')
        return -1
    print(f'{bcolors.OKBLUE}Enter your option:{bcolors.ENDC}', end='')
    return 0

def print_top_k_relations(entityId, relations_arr):
    print(f'{bcolors.OKGREEN}====================== TOP-K relations ======================{bcolors.ENDC}')
    for relation in relations_arr:
        # if relation[0] == 'http://www.wikidata.org/prop/direct/P31':
        #     print(f' {bcolors.WARNING}===================== Skip P31 ====================={bcolors.ENDC}')
        #     continue
        print(f'{bcolors.BOLD} {relation[0]} ---- {relation[1]} {bcolors.ENDC}', end='\n')
    print(f'{bcolors.OKGREEN}============================ End ============================{bcolors.ENDC}', end='\n')


    amountOfSamples = input('Max number of samples to be predicted for these relations: ')
    download_wikipedia_articles_predict()
    generate_predict_samples(entityId, relations_arr, amountOfSamples)

def print_info():
    print(f'{bcolors.OKGREEN}====================== INFO ======================{bcolors.ENDC}', end='\n')
    print(f'{bcolors.WARNING}0 - Run the option 3 in main menu in the first time. [IMPORTANT] {bcolors.ENDC}', end='\n')
    print(f'{bcolors.WARNING}1 - The pre-trained|base model should be put in the ./base_model folder{bcolors.ENDC}', end='\n')
    print(f'{bcolors.OKGREEN}====================== END ======================{bcolors.ENDC}', end='\n')
    input('[ENTER] to continue')

"""
    ============================= END PRINT MENUS ==============================
"""

"""
                Main function
"""
def main():
    while (True):
        if print_menu() != -1:
            option = input()
            if (option == '1'):
                train_pipeline()
            elif (option == '2'):
                predict_newfacts()
            elif (option == '3'):
                create_all_folders()
            elif(option == '4'):
                print_info()
            elif (option == '0'):
                break;
            else:
                print(f'{bcolors.FAIL}Wrong input, enter option between 1~3{bcolors.ENDC}')
        else:
            break

if __name__ == '__main__':
    main()
