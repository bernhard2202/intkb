import os
import argparse
import json

def main():
    print('-'*55)
    print(f'{args.dir_eval:^55}')
    print('-'*55)
    list_iters = [f for f in os.listdir(os.path.join(args.dir_eval)) if f.startswith('iter_')]
    total = 0
    for i in range(40):
        if os.path.exists(os.path.join(args.dir_eval, 'iter_{}/log.txt'.format(i))):
            with open(os.path.join(args.dir_eval, 'iter_{}/log.txt'.format(i)), 'r') as f:
                text = f.read()
                len_iter = int(text.split(' ')[-6])
                total += len_iter
                iter_n = str(i) if i > 9 else ('_'+str(i))                
                print(f'ITER_{iter_n} >>  POS/NEG ({int(len_iter/2 + len_iter/2):^5}) added to the DS_train.json')
    if os.path.exists(os.path.join(args.dir_eval, 'DS_train.json')):
        with open(os.path.join(args.dir_eval, 'DS_train.json'), 'r') as f:
            res = json.load(f)
    print('-'*55)
    print(f'              All adds up >>> {total} samples')
    print('\n            DS_train.json >>> {} samples\n{}'.format(len(res), '-'*55))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='print logs')
    parser.add_argument('--dir_eval', type=str, default='combined_1',  help='path to dir to evaluate')
    args = parser.parse_args()
    main()
