import os
import json
import torch
from torch.utils.data import Dataset, TensorDataset
from feature import convert_examples_to_features

def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False

class SquadExample():
    def __init__(self, qid, context, question, answer, start, end, is_impossible):
        self.qid = qid
        self.context = context
        self.question = question
        self.answer = answer
        self.start = start
        self.end = end
        self.is_impossible = is_impossible
        
    def __repr__(self):
        #return self.context[self.start:self.end]
        #if self.context[self.start:self.end] != self.answer:
        #    return 'NA!! {} - {}'.format(self.context[self.start:self.end], answer)
        return 'id:{}  question:{}...  answer:{}...  is_impossible:{}'.format(
            self.qid,
            self.question[:10],
            self.answer[:10],
            self.is_impossible)

class SquadDataset(Dataset):
    def __init__(self, path, tokenizer, is_train=True, is_inference=False):
        '''
        path: SquadDataset 데이터셋 위치
        tokenizer: Squad 데이터셋을 토크나이징할 토크나이저, ex) BertTokenizer
        is_train: SquadDataset을 정의하는 목적이 모델 학습용일 경우 True, 그렇지 않으면 False
        is_inference: SquadDataset을 정의하는 목적이 인퍼런스용일 경우 True, 그렇지 않으면 False
        '''
        
        if is_train:
            filename = os.path.join(path, 'train-v2.0.json')
        else:
            if is_inference:
                filename = os.path.join(path, 'test-v2.0.json')
            else:
                filename = os.path.join(path, 'dev-v2.0.json')

        cached_features_file = os.path.join(os.path.dirname(filename), 'cached_{}_64.cache'.format('train' if is_train else 'valid'))
        #cached_examples_file = os.path.join(os.path.dirname(filename), 'cached_example_{}_64.cache'.format('train' if is_train else 'valid'))

        if os.path.exists(cached_features_file):
            print('cache file exists')
            #self.examples = torch.load(cached_examples_file)
            self.features = torch.load(cached_features_file)
        else:
            print('cache file does not exist')

            with open(filename, "r", encoding='utf-8') as reader:
                input_data = json.load(reader)["data"]

            self.examples = []
            for entry in input_data:
                for paragraph in entry["paragraphs"]:
                    context = paragraph['context']
                    
                    doc_tokens = []
                    char_to_word_offset = []
                    prev_is_whitespace = True
                    for c in context:
                        if is_whitespace(c):
                            prev_is_whitespace = True
                        else:
                            if prev_is_whitespace:
                                doc_tokens.append(c)
                            else:
                                doc_tokens[-1] += c
                            prev_is_whitespace = False
                        char_to_word_offset.append(len(doc_tokens) - 1)
                            
                            
                    for qa in paragraph['qas']:
                        is_impossible = qa['is_impossible']
                        
                        if not is_impossible:
                            answer = qa['answers'][0]
                            original_answer = answer['text']
                            answer_start = answer['answer_start']
                            
                            answer_length = len(original_answer)
                            start_pos = char_to_word_offset[answer_start]
                            end_pos = char_to_word_offset[answer_start + answer_length - 1]

                            answer_end = answer_start + len(original_answer)
                        else:
                            original_answer = ''
                            start_pos = 1
                            end_pos = -1

                        example = SquadExample(
                            qid=qa['id'],
                            context=doc_tokens,
                            question=qa['question'],
                            answer=original_answer,
                            start=start_pos,
                            end=end_pos,
                            is_impossible=is_impossible)
                        self.examples.append(example)
            print('examples: {}'.format(len(self.examples)))

            self.features = convert_examples_to_features(
                examples=self.examples,
                tokenizer=tokenizer,
                max_seq_length=384,
                doc_stride=128,
                max_query_length=64,
                is_training=True if not is_inference else False)
            print('is_training: {}'.format(True if not is_inference else False))

            # torch.save(self.examples, cached_examples_file)
            torch.save(self.features, cached_features_file)

        '''
        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in self.features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in self.features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in self.features], dtype=torch.long)
        all_cls_index = torch.tensor([f.cls_index for f in self.features], dtype=torch.long)
        all_p_mask = torch.tensor([f.p_mask for f in self.features], dtype=torch.float)
        if is_train:
            all_start_positions = torch.tensor([f.start_position for f in self.features], dtype=torch.long)
            all_end_positions = torch.tensor([f.end_position for f in self.features], dtype=torch.long)
            dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                    all_start_positions, all_end_positions,
                                    all_cls_index, all_p_mask)
        else:
            all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
            dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_example_index, all_cls_index, all_p_mask)

        return dataset
        '''


    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]
