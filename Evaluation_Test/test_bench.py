import torch
from videollava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from videollava.conversation import conv_templates, SeparatorStyle
from videollava.model.builder import load_pretrained_model
from videollava.utils import disable_torch_init
from videollava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from torchvision import transforms
import json
from tqdm import tqdm
import os
import argparse



from PIL import Image

import random
import numpy as np
from torch.utils.data import Dataset


def get_prompt2(conv):
    ret = conv.system + conv.sep
    count = 0
    for role, message in conv.messages:
        count += 1
        if count == len(conv.messages):
            ret += role + ": " + message
        else:
            if message:
                ret += role + ": " + message + conv.sep
            else:
                ret += role + ":"
    return ret

class MLVU(Dataset):
    def __init__(self, data_dir, video_folder):
        self.video_folder=video_folder
        self.data_list = []
        with open(os.path.join(data_dir), 'r') as f:
            json_data = json.load(f)
        for data in json_data:
            self.data_list.append({
                    'task_type': data["question_type"],
                    'data': data,
                    'question_id':data["question_id"]
                })
    
    def __str__(self):
        len_list = {}
        option_list = {}
        for data in self.data_list:
            if data['task_type'] not in len_list:
                len_list[data['task_type']] = 0
            len_list[data['task_type']] += 1
            if data['task_type'] not in option_list:
                option_list[data['task_type']] = 0
            option_list[data['task_type']] += len(data['data']['candidates'])
        
        correct = 0
        total = 0
        res = f"There are {len(self.data_list)} videos as follow:\n"
        for k, v in len_list.items():
            correct += len_list[k]
            total += option_list[k]
            res += f"{v} for {k} ({option_list[k]} options => {len_list[k]/option_list[k]*100:.2f}%)\n"
            correct = correct + 1 / option_list[k]
        res += f"Total random accuracy: {correct/total*100:.2f}%"
        return res.rstrip()
        
    def __len__(self):
        return len(self.data_list)
    
    def get_index(self, bound, fps, max_frame, first_idx=0):
        start, end = -100000, 100000
        start_idx = max(first_idx, round(start * fps))
        end_idx = min(round(end * fps), max_frame)
        seg_size = float(end_idx - start_idx) / self.num_segments
        frame_indices = np.array([
            int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
            for idx in range(self.num_segments)
        ])
        return frame_indices
    

    def qa_template(self, data):
        question = f"Question: {data['question']}\n"
        question += "Options:\n"
       
        for idx, c in enumerate(data['candidates']):
            question += f"({chr(ord('A') + idx)}) {c}\n"

        question = question.rstrip()
 
        return question

    def __getitem__(self, idx):
        bound = None
        video_path = os.path.join(self.video_folder, self.data_list[idx]['data']['video'])
        question = self.qa_template(self.data_list[idx]['data'])
        return {
            'video': video_path, 
            'question': question, 
            'task_type': self.data_list[idx]['task_type'],
            "question_id":self.data_list[idx]["question_id"]
        }





def main():
    disable_torch_init()


    video_folder="/Evaluation_LVBench/MLVU_Test/video"
    data_dir = f"/Evaluation_LVBench/MLVU_Test/leaderboard/test_question.json"



    save_path = f"./test_all_choice"
    result_path=f"bench_all.json"

    dataset = MLVU(data_dir, video_folder)

    model_path = '/share/junjie/shuyan/Video-LLaVA/LanguageBind/Video-LLaVA-7B'
    cache_dir = 'cache_dir'
    device = 'cuda:6'
    load_4bit, load_8bit = True, False
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, processor, _ = load_pretrained_model(model_path, None, model_name, load_8bit, load_4bit, device=device)
    video_processor = processor['video']
    conv_mode = "llava_v1"
    conv = conv_templates[conv_mode].copy()
    roles = conv.roles


    correct = 0
    total = 0
    res_list = []
    acc_dict = {}
    for example in tqdm(dataset):
        conv.messages = list()
        task_type = example['task_type']
        if task_type not in acc_dict:
            acc_dict[task_type] = [0, 0] # correct, total
        acc_dict[task_type][1] += 1
        total += 1
        video_path=example["video"]
        inp=example["question"] + "\nOnly give the best option."


        video_tensor = video_processor(video_path, return_tensors='pt')['pixel_values']
        print("##########",video_tensor.shape)
      
        if type(video_tensor) is list:
            tensor = [video.to(model.device, dtype=torch.float16) for video in video_tensor]
        else:
            tensor = video_tensor.to(model.device, dtype=torch.float16)

        inp = ' '.join([DEFAULT_IMAGE_TOKEN] * model.get_video_tower().config.num_frames) + '\n' + inp
        conv.system="Carefully watch this video and pay attention to every detail. Based on your observations, select the best option that accurately addresses the question."
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], "Best Option: (")
        # prompt = conv.get_prompt()
        prompt=get_prompt2(conv)
        print("*************")
        print("prompt",prompt)
        print("**************")
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=tensor,
                do_sample=True,
                temperature=0.1,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=[stopping_criteria])

        pred= tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()

        if (")") in pred:
            index=pred.index(")")
            pred=pred[:index].strip()
        print("Pred",pred)

     

        res_list.append({
        'question_id': example["question_id"],
        'question_type':example['task_type'],
        'option':pred
        })
    
    with open(f"test_res.json", "w") as f:
        json.dump(res_list, f)


if __name__ == '__main__':
    main()