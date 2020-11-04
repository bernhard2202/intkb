import os
import json

for split in ['known', 'zero_shot']:
    print(split)
    train_relations = [f for f in os.listdir(os.path.join('data', 'splits', split))
                       if f.endswith('_train.json') and f.startswith('P')]
    test_relations = [f for f in os.listdir(os.path.join('data', 'splits', split))
                      if f.endswith('_test.json') and f.startswith('P')]

    no_test = 0
    no_train = 0

    head_ent = set()
    tail_ent = set()
    print(len(test_relations))
    assert len(test_relations) == len(train_relations)

    for train_relation in train_relations:
        with open(os.path.join('data', 'splits', split, '{}'.format(train_relation)), 'r') as f:
            jdata = json.load(f)

        no_train += len(jdata)
        for dat in jdata:
            for ent in dat['answer_entity']:
                tail_ent.add(ent)
            head_ent.add(dat['wikipedia_link'])

    for test_relation in test_relations:
        with open(os.path.join('data', 'splits', split, '{}'.format(test_relation)), 'r') as f:
            jdata = json.load(f)

        no_test += len(jdata)
        for dat in jdata:
            for ent in dat['answer_entity']:
                tail_ent.add(ent)
            head_ent.add(dat['wikipedia_link'])

    print(len(head_ent))
    print(len(tail_ent))
    print(no_train)
    print(no_test)
