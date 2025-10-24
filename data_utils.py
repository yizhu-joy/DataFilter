import numpy as np
from copy import deepcopy
import io, json
from config import IGNORE_ATTACK_SENTENCES, OTHER_DELM_FOR_TEST, OTHER_DELM_TOKENS, IGNORE_INDEX, DELIMITERS, CONTEXT_ATTACK_SENTENCES


def format_with_other_delimiters(text, test=False):
    test_idx = - OTHER_DELM_FOR_TEST
    mark = np.random.choice(OTHER_DELM_TOKENS['mark'][test_idx:] if test else OTHER_DELM_TOKENS['mark'][:test_idx]) + ':'
    
    def sample_delm(delm_name):
        role_name = 'user' if (delm_name == 'inst' or delm_name == 'inpt') else 'asst'
        if test: 
            role = np.random.choice(OTHER_DELM_TOKENS[role_name][test_idx:]) 
            delm = np.random.choice(OTHER_DELM_TOKENS[delm_name][test_idx:])
        else:    
            role = np.random.choice(OTHER_DELM_TOKENS[role_name][:test_idx]) 
            delm = np.random.choice(OTHER_DELM_TOKENS[delm_name][:test_idx])
        
        p = np.random.rand()
        if p < 1/3: return (role + delm).upper()
        elif p < 2/3: return (role + delm).lower()
        else: return role + delm
    
    for delm in DELIMITERS.values():
        if '' in delm or ' ' in delm: continue
        text = text.replace(delm[0], mark.format(s=sample_delm('inst')))
        text = text.replace(delm[1], mark.format(s=sample_delm('inpt')))
        text = text.replace(delm[2], mark.format(s=sample_delm('resp')))
    return text

def random_insert(input, injection):
    if injection.strip() == '': return input
    input_words = input.split()
    if len(input_words) == 0: return injection
    insert_pos = np.random.randint(0, len(input_words)+1)
    new_words = ' '.join(input_words[:insert_pos]) + ' ' + injection + ' ' + ' '.join(input_words[insert_pos:])
    return new_words.strip()
 

def generate_training_data(data_dicts, attack, random_position=False, cut_benign=False, train_or_test='train'):
    """
       Generate adversarial training data with prompt injection attacks.
       
       Args:
           data_dicts: List of dicts with 'instruction', 'input', 'output' keys
           attack: Attack type - 'None', 'Naive', 'Ignore', 'Context', 'Completion', etc.
           random_position: If True, randomly place injections (prepend/append/random), else always append
           cut_benign: If True, truncate benign data
           train_or_test: 'train' or 'test' - determines attack sentence pool
           
       Returns:
           tuple: (targets, sources, injections, inject_tasks, positions, outputs, instructions)
       """
    if attack == 'None':
        return [
            example.get("instruction", "") + " <|end_of_instruction|> " + example.get("input", "") + " <|end_of_data|> " for example in data_dicts if example.get("input", "") != ""
        ], [
            example.get("instruction", "") + " <|end_of_instruction|> "  +  example.get("input", "")  +  " <|end_of_data|> " for example in data_dicts if example.get("input", "") != "" 
        ], ["" for example in data_dicts if example.get("input", "") != ""], ["" for example in data_dicts if example.get("input", "") != ""], ["" for example in data_dicts if example.get("input", "") != ""], [f"{example['output']}" for example in data_dicts if example.get("input", "") != ""], [example.get("instruction", "") for example in data_dicts if example.get("input", "") != ""]

    if attack == 'Completion' or attack == 'CompletionIgnore' or attack == 'StrongDelimiterCompletion' or attack == 'MultishotCompletion':
        ref_inst_resp = {}
        for ref_sample in jload('data/alpaca_data_cleaned.json'):  ref_inst_resp[ref_sample['instruction']] = ref_sample['output']
    
    targets = []
    sources = []
    injections = []
    inject_tasks = []
    positions = []
    outputs_davinci_003 = []
    instructions = []

    for i in range(len(data_dicts)):
        # Only keep the example with input, return the example's input (not format_map) and output
        if data_dicts[i].get("input", "") != "":
            injected_sample = deepcopy(np.random.choice(data_dicts)) 
            if injected_sample['instruction'][-1] == '?' or injected_sample['instruction'][-2:] == '? ':
                injected_prompt = 'answer the following question. ' + injected_sample['instruction'] + ' ' + injected_sample['input']
            else: 
                injected_prompt = injected_sample['instruction'][0].lower() + injected_sample['instruction'][1:] + ' ' + injected_sample['input']
                if injected_prompt[-1] not in ['.', '!', '?'] and injected_prompt[-2:] not in ['. ', '! ', '? ']: injected_prompt += '.'  


            data_dicts_item = deepcopy(data_dicts[i])

            position = 'append' # default
            if random_position:
                p = np.random.rand()
                # add random position for injection. 20% append injection, 20% prepend injection, 60% random position
                if p < 0.2: position = 'append'
                elif p < 0.4: position = 'prepend'
                else: position = 'random'

            if cut_benign:
                # randomly truncate the benign data. 15% remove all, 10% keep 2/3, 10% keep 1/2
                p_c = np.random.rand()
                if p_c < 0.15:
                    data_dicts_item['input'] = ''
                elif p_c < 0.25:
                    data_dicts_item['input'] = data_dicts_item['input'][:int(len(data_dicts_item['input'])*(2/3))].strip()
                elif p_c < 0.35:
                    data_dicts_item['input'] = data_dicts_item['input'][:int(len(data_dicts_item['input'])*(1/2))].strip()

            benign_data = data_dicts_item['input']

            if attack == 'Naive':
                if position == 'prepend':
                    data_dicts_item['input'] = injected_prompt[0].upper() + injected_prompt[1:] + ' ' + data_dicts_item['input']
                elif position == 'append':
                    data_dicts_item['input'] += ' ' + injected_prompt[0].upper() + injected_prompt[1:]
                elif position == 'only':
                    data_dicts_item['input'] = injected_prompt[0].upper() + injected_prompt[1:]
                elif position == 'random':
                    data_dicts_item['input'] = random_insert(data_dicts_item['input'], injected_prompt[0].upper() + injected_prompt[1:])
                else: raise NotImplementedError
                data_dicts_item['injection'] = injected_prompt[0].upper() + injected_prompt[1:]
                data_dicts_item['inject_task'] = injected_prompt[0].upper() + injected_prompt[1:]
            elif attack == 'Ignore':
                # if train_or_test == 'train', use the training ignore sentences
                # if train_or_test == 'test', use the test ignore sentences
                data_dicts_item['injection'] = np.random.choice(IGNORE_ATTACK_SENTENCES[train_or_test]).format(injected_prompt=injected_prompt)
                if position == 'prepend':
                    data_dicts_item['input'] = data_dicts_item['injection'] + ' ' + data_dicts_item['input']
                elif position == 'append':
                    data_dicts_item['input'] += ' ' + data_dicts_item['injection']
                elif position == 'only':
                    data_dicts_item['input'] = data_dicts_item['injection']
                elif position == 'random':
                    data_dicts_item['input'] = random_insert(data_dicts_item['input'], data_dicts_item['injection'])
                else: raise NotImplementedError
                data_dicts_item['inject_task'] = injected_prompt[0].upper() + injected_prompt[1:]
            elif attack == 'Context':
                task_key_word = data_dicts_item['task_key_word'] 
                data_dicts_item['injection'] = np.random.choice(CONTEXT_ATTACK_SENTENCES[train_or_test]).format(injected_prompt=injected_prompt, task_key_word=task_key_word)
                if position == 'prepend':
                    data_dicts_item['input'] = data_dicts_item['injection'] + ' ' + data_dicts_item['input']
                elif position == 'append':
                    data_dicts_item['input'] += ' ' + data_dicts_item['injection']
                elif position == 'only':
                    data_dicts_item['input'] = data_dicts_item['injection']
                elif position == 'random':
                    data_dicts_item['input'] = random_insert(data_dicts_item['input'], data_dicts_item['injection'])
                else: raise NotImplementedError
                data_dicts_item['inject_task'] = injected_prompt[0].upper() + injected_prompt[1:]
            elif attack == 'Completion':
                data_dicts_item['injection'] = '\n\n' + DELIMITERS['SpclSpclSpcl'][2] + '\n' + ref_inst_resp.get(data_dicts_item['instruction'], data_dicts_item['output']) + '\n\n' + \
                                                    DELIMITERS['SpclSpclSpcl'][0] + '\n' + injected_prompt.capitalize()
                if injected_sample['input'] != '':
                    data_dicts_item['injection'] += '\n\n' + DELIMITERS['SpclSpclSpcl'][1] + '\n' + injected_sample['input']
                data_dicts_item['injection'] = format_with_other_delimiters(data_dicts_item['injection'], test=(train_or_test=='test'))
                if position == 'prepend':
                    data_dicts_item['input'] = data_dicts_item['injection'] + ' ' + data_dicts_item['input']
                elif position == 'append':
                    data_dicts_item['input'] += data_dicts_item['injection']
                elif position == 'only':
                    data_dicts_item['input'] = data_dicts_item['injection']
                elif position == 'random':
                    data_dicts_item['input'] = random_insert(data_dicts_item['input'], data_dicts_item['injection'])
                else: raise NotImplementedError
                data_dicts_item['inject_task'] = injected_prompt[0].upper() + injected_prompt[1:]
                
            elif attack == 'CompletionIgnore':
                ignore_instruction = np.random.choice(IGNORE_ATTACK_SENTENCES[train_or_test]).format(injected_prompt=injected_prompt)
                data_dicts_item['injection'] = '\n\n' + DELIMITERS['SpclSpclSpcl'][2] + '\n' + ref_inst_resp.get(data_dicts_item['instruction'], data_dicts_item['output']) + '\n\n' + \
                                                    DELIMITERS['SpclSpclSpcl'][0] + '\n' + ignore_instruction[0].upper() + ignore_instruction[1:]
                if injected_sample['input'] != '':
                    data_dicts_item['injection'] += '\n\n' + DELIMITERS['SpclSpclSpcl'][1] + '\n' + injected_sample['input']
                data_dicts_item['injection'] = format_with_other_delimiters(data_dicts_item['injection'], test=(train_or_test=='test'))
                if position == 'prepend':
                    data_dicts_item['input'] = data_dicts_item['injection'] + ' ' + data_dicts_item['input']
                elif position == 'append':
                    data_dicts_item['input'] += data_dicts_item['injection']
                elif position == 'only':
                    data_dicts_item['input'] = data_dicts_item['injection']
                elif position == 'random':
                    data_dicts_item['input'] = random_insert(data_dicts_item['input'], data_dicts_item['injection'])
                else: raise NotImplementedError
                data_dicts_item['inject_task'] = injected_prompt[0].upper() + injected_prompt[1:]

            elif attack == 'StrongDelimiterCompletion':
                data_dicts_item['injection'] = '\n\n' + "alth>assistantalth" + '\n' + ref_inst_resp.get(data_dicts_item['instruction'], data_dicts_item['output']) + '\n\n' + \
                                                    "alth>useralth" + '\n' + injected_prompt.capitalize()
                if injected_sample['input'] != '':
                    data_dicts_item['injection'] += '\n\n' + DELIMITERS['SpclSpclSpcl'][1] + '\n' + injected_sample['input']
                if position == 'prepend':
                    data_dicts_item['input'] = data_dicts_item['injection'] + ' ' + data_dicts_item['input']
                elif position == 'append':
                    data_dicts_item['input'] += data_dicts_item['injection']
                elif position == 'only':
                    data_dicts_item['input'] = data_dicts_item['injection']
                elif position == 'random':
                    data_dicts_item['input'] = random_insert(data_dicts_item['input'], data_dicts_item['injection'])
                else: raise NotImplementedError
                data_dicts_item['inject_task'] = injected_prompt[0].upper() + injected_prompt[1:]

            elif attack == 'MultishotCompletion':
                n_shot = 2 #(by default)
                # Phase 1: complete the user's instruction
                data_dicts_item['injection'] = '\n\n' + DELIMITERS['SpclSpclSpcl'][2] + '\n' + ref_inst_resp.get(data_dicts_item['instruction'], data_dicts_item['output']) + '\n\n' + \
                                                    DELIMITERS['SpclSpclSpcl'][0] + '\n'
                # Phase 2: add n_shot examples
                for _ in range(n_shot):
                    multi_shot_injected_sample = deepcopy(np.random.choice(data_dicts))
                    multi_shot_injected_sample['instruction'] = multi_shot_injected_sample['instruction']
                    if multi_shot_injected_sample['instruction'][-1] == '?' or multi_shot_injected_sample['instruction'][-2:] == '? ':
                        multi_shot_injected_prompt = 'answer the following question. ' + multi_shot_injected_sample['instruction'] + ' ' + multi_shot_injected_sample['input']
                    else:
                        multi_shot_injected_prompt = multi_shot_injected_sample['instruction'][0].lower() + multi_shot_injected_sample['instruction'][1:] + ' ' + multi_shot_injected_sample['input']
                        if multi_shot_injected_prompt[-1] not in ['.', '!', '?'] and multi_shot_injected_prompt[-2:] not in ['. ', '! ', '? ']: multi_shot_injected_prompt += '.'  
                    data_dicts_item['injection'] += multi_shot_injected_prompt.capitalize() + '\n\n' + DELIMITERS['SpclSpclSpcl'][1] + '\n' + multi_shot_injected_sample['input'] + '\n\n' + DELIMITERS['SpclSpclSpcl'][2] + '\n' + ref_inst_resp.get(multi_shot_injected_sample['instruction'], multi_shot_injected_sample['output']) + '\n\n' + DELIMITERS['SpclSpclSpcl'][0] + '\n'
                # Phase 3: add the injected task
                data_dicts_item['injection'] += injected_prompt.capitalize()
                if injected_sample['input'] != '':
                    data_dicts_item['injection'] += '\n\n' + DELIMITERS['SpclSpclSpcl'][1] + '\n' + injected_sample['input']
                data_dicts_item['injection'] = format_with_other_delimiters(data_dicts_item['injection'], test=(train_or_test=='test'))
                if position == 'prepend':
                    data_dicts_item['input'] = data_dicts_item['injection'] + ' ' + data_dicts_item['input']
                elif position == 'append':
                    data_dicts_item['input'] += data_dicts_item['injection']
                elif position == 'only':
                    data_dicts_item['input'] = data_dicts_item['injection']
                elif position == 'random':
                    data_dicts_item['input'] = random_insert(data_dicts_item['input'], data_dicts_item['injection'])
                else: raise NotImplementedError
                data_dicts_item['inject_task'] = injected_prompt[0].upper() + injected_prompt[1:]
            else: raise NotImplementedError
            targets.append(data_dicts_item['instruction'] + " <|end_of_instruction|> " + (benign_data) + " <|end_of_data|> ")
            sources.append(data_dicts_item['instruction'] + " <|end_of_instruction|> " + (data_dicts_item['input'])  +  " <|end_of_data|> ") 
            injections.append(data_dicts_item['injection'])
            inject_tasks.append(data_dicts_item['inject_task'])
            positions.append(position)
            outputs_davinci_003.append(data_dicts_item['output'])
            instructions.append(data_dicts_item['instruction'])
    print("Created dataset with", len(sources), "examples")
    return targets, sources, injections, inject_tasks, positions, outputs_davinci_003, instructions


def jload(f, mode="r"):
    if not isinstance(f, io.IOBase): f = open(f, mode=mode)
    jdict = json.load(f)
    f.close()
    return jdict

def jdump(obj, f, mode="w", indent=4, default=str):
    if not isinstance(f, io.IOBase): f = open(f, mode=mode)
    if isinstance(obj, (dict, list)): json.dump(obj, f, indent=indent, default=default)
    elif isinstance(obj, str): f.write(obj)
    else: raise ValueError(f"Unexpected type: {type(obj)}")
    f.close()

