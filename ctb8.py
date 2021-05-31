# -*- coding:utf-8 -*-
# Filename: make_ctb.py
# 切分：https://bbs.hankcs.com/t/topic/3024
# 开源：https://wakespace.lib.wfu.edu/handle/10339/39379
# Author：hankcs
import argparse
from os import listdir
from os.path import isfile, join, isdir
import errno
from os import makedirs
import sys
import nltk
import csv


def make_sure_path_exists(path):
    try:
        makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def combine_files(fids, out, tb, task, add_s=False):
    print('%d files...' % len(fids))
    total_sentence = 0
    for n, file in enumerate(fids):
        if n % 10 == 0 or n == len(fids) - 1:
            print("%c%.2f%%" % (13, (n + 1) / float(len(fids)) * 100), end='')
        sents = tb.parsed_sents(file)
        for s in sents:
            if task == 'par':
                if add_s:
                    out.write('(S {})'.format(s.pformat(margin=sys.maxsize)))
                else:
                    out.write(s.pformat(margin=sys.maxsize))
            elif task == 'pos':
                for word, tag in s.pos():
                    if tag == '-NONE-':
                        continue
                    out.write('{}\t{}\n'.format(word, tag))
            elif task == 'pos-pku':
                for word, tag in s.pos():
                    if tag == '-NONE-':
                        continue
                    out.write('{}/{} '.format(word, tag))
            elif task == 'seg':
                for word, tag in s.pos():
                    if tag == '-NONE-':
                        continue
                    if len(word) == 1:
                        out.write(word + "\tS\n")
                    else:
                        out.write(word[0] + "\tB\n")
                        for w in word[1:len(word) - 1]:
                            out.write(w + "\tM\n")
                        out.write(word[len(word) - 1] + "\tE\n")
            else:
                raise RuntimeError('Invalid task {}'.format(task))
            out.write('\n')
            total_sentence += 1
    print()
    print('%d sentences.' % total_sentence)
    print()


def convert_ctb8_to_bracketed(ctb_root, out_root):
    ctb_root = join(ctb_root, 'bracketed')
    chtbs = [f for f in listdir(ctb_root) if isfile(join(ctb_root, f)) and f.startswith('chtb')]
    make_sure_path_exists(out_root)
    for f in chtbs:
        with open(join(ctb_root, f), encoding='utf-8') as src, open(join(out_root, f + '.txt'), 'w', encoding='utf-8') as out:
            for line in src:
                if not line.startswith('<'):
                    out.write(line)


def split(ctb_root):
    chtbs = [f for f in listdir(ctb_root) if isfile(join(ctb_root, f)) and f.startswith('chtb')]
    folder = {}
    for f in chtbs:
        tag = f[-6:-4]
        if tag not in folder:
            folder[tag] = []
        folder[tag].append(f)
    train, dev, test = [], [], []
    for tag, files in folder.items():
        t = int(len(files) * .8)
        d = int(len(files) * .9)
        train += files[:t]
        dev += files[t:d]
        test += files[d:]
    return train, dev, test


def combine_fids(fids, out_path, task):
    print('Generating ' + out_path)
    files = []
    for f in fids:
        if isfile(join(ctb_in_nltk, f)):
            files.append(f)
    with open(out_path, 'w', encoding='utf-8') as out:
        combine_files(files, out, ctb, task, add_s=True)


def find_nltk_data():
    global ctb_in_nltk
    for root in nltk.data.path:
        if isdir(root):
            ctb_in_nltk = root


def extract(path):
    data = []
    with open(path) as f:  # 共1342249行，55999句
        tsvreader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
        sent = []
        tags = []
        for line in tsvreader:
            if line != []:
                sent.append(line[0])
                tags.append(line[1])
            else:
                text = ''.join(sent)
                target = [i + '_' + e for i, e in zip(sent, tags)]
                target = '/'.join(target)
                data.append([text, target])
                sent, tags = [], []
    return data


def write(path,data):
    with open(path, "w") as csvfile:  #encoding='utf-8',newline=''
        writer = csv.writer(csvfile)
        writer.writerow(["src", "tgt"])
        writer.writerows(data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Combine Chinese Treebank 8 bracketed files into train/dev/test set')
    parser.add_argument("--ctb", default='/home/di/Desktop/data/ctb8.0/data',
                        help='The root path to Chinese Treebank 8')
    parser.add_argument("--output", default='/home/di/Desktop/thesis/ctb',
                        help='The folder where to store the output train.txt/dev.txt/test.txt')
    parser.add_argument("--task", dest="task", default='pos',
                        help='Which task (seg, pos, par)? Use seg for word segmentation, pos for part-of-speech '
                             'tagging (pos-pku for PKU format, otherwise tsv format), par for phrase structure parsing')

    args = parser.parse_args()

    ctb_in_nltk = None
    find_nltk_data()

    if ctb_in_nltk is None:
        nltk.download('ptb')
        find_nltk_data()

    ctb_in_nltk = join(ctb_in_nltk, 'corpora')
    ctb_in_nltk = join(ctb_in_nltk, 'ctb8')

    if not isdir(ctb_in_nltk):
        print('Converting CTB: removing xml tags...')
        convert_ctb8_to_bracketed(args.ctb, ctb_in_nltk)
    print('Importing to nltk...\n')
    from nltk.corpus import BracketParseCorpusReader, LazyCorpusLoader

    ctb = LazyCorpusLoader(
        'ctb8', BracketParseCorpusReader, r'chtb_.*\.txt',
        tagset='unknown')

    training, development, test = split(ctb_in_nltk)
    task = args.task
    if task == 'par' or task == 'pos-pku':
        ext = 'txt'
    elif task == 'seg' or task == 'pos':
        ext = 'tsv'
    else:
        eprint('Invalid task {}'.format(task))
        exit(1)

    root_path = args.output
    make_sure_path_exists(root_path)
    combine_fids(training, join(root_path, 'train.{}'.format(ext)), task)
    combine_fids(development, join(root_path, 'dev.{}'.format(ext)), task)
    combine_fids(test, join(root_path, 'test.{}'.format(ext)), task)
