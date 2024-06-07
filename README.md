<h1 align="center">MLVU: Multi-task Long Video Understanding Benchmark</h1>
<p align="center">
    <a href="https://arxiv.org/abs/2406.04264">
            <img alt="Build" src="http://img.shields.io/badge/cs.CV-arXiv%3A2406.04264-B31B1B.svg">
    </a>
    <a href="https://huggingface.co/datasets/MLVU/MVLU">
        <img alt="Build" src="https://img.shields.io/badge/🤗 Dataset-MLVU Benchmark-yellow">
    </a>
</p>

This repo contains the annotation data and evaluation code for the paper "[MLVU: A Comprehensive Benchmark for Multi-Task Long Video Understanding](https://arxiv.org/abs/2406.04264)".

## :bell: News:
- 🥳 6/7/2024: We have released the MLVU [Benchmark](https://huggingface.co/datasets/MLVU/MVLU) and [Paper](https://arxiv.org/abs/2406.04264)! :fire:



## Introduction
We introduce MLVU: the first comprehensive benchmark designed for evaluating Multimodal Large Language Models (MLLMs) in Long Video Understanding (LVU) tasks. MLVU is constructed from a wide variety of long videos, with lengths ranging from 3 minutes to 2 hours, and includes nine distinct evaluation tasks. These tasks challenge MLLMs to handle different types of tasks, leveraging both global and local information from videos. Our evaluation of 20 popular MLLMs, including GPT-4o, reveals significant challenges in LVU, with even the top-performing GPT-4o only achieving an average score of 64.6% in multi-choice tasks. In addition, our empirical results underscore the need for improvements in context length, image understanding, and strong LLM-backbones. We anticipate that MLVU will serve as a catalyst for the community to further advance MLLMs' capabilities in understanding long videos.

![Statistical overview of our LVBench dataset. **Left:** Distribution of Video Duration; **Middle** Distribution of Source Types for Long Videos; **Right:** Quantification of Each Task Type.](./figs/statistic.png)



## :trophy: Mini-Leaderboard










## License
Our dataset is under the CC-BY-NC-SA-4.0 license.

:warning: If you need to access and use our dataset, you must understand and agree: **This dataset is for research purposes only and cannot be used for any commercial or other purposes. The user assumes all effects arising from any other use and dissemination.**

We do not own the copyright of any raw video files. Currently, we provide video access to researchers under the condition of acknowledging the above license. For the video data used, we respect and acknowledge any copyrights of the video authors. Therefore, for the movies, TV series, documentaries, and cartoons used in the dataset, we have reduced the resolution, clipped the length, adjusted dimensions, etc. of the original videos to minimize the impact on the rights of the original works. 

If the original authors of the related works still believe that the videos should be removed, please contact mlvubenchmark@gmail.com or directly raise an issue.


## MLVU Benchmark
> Before you access our dataset, we kindly ask you to thoroughly read and understand the license outlined above. If you cannot agree to these terms, we request that you refrain from downloading our video data.


The annotation file is readily accessible [here](url). For the raw videos, you can access them via this [<u>🤗 HF Link</u>](https://huggingface.co/datasets/MLVU/MVLU).


MLVU encompasses nine distinct tasks, which include multiple-choice tasks as well as free-form generation tasks. These tasks are specifically tailored for long-form video understanding, and are classified into three categories: holistic understanding, single detail understanding, and multi-detail understanding. Examples of the tasks are displayed below.


![Task Examples of our MLVU.](./figs/task_example.png)


## Evaluation
Please refer to our eval folder for more details.




## Hosting and Maintenance
The annotation files will be permanently retained. 

If some videos are requested to be removed, we will replace them with a set of video frames sparsely sampled from the video and adjusted in resolution. Since **all the questions in MLVU are only related to visual content** and do not involve audio, this will not significantly affect the validity of MLVU (most existing MLLMs also understand videos by frame extraction).

If even retaining the frame set is not allowed, we will still keep the relevant annotation files, and replace them with the meta-information of the video, or actively seek more reliable and risk-free video sources.


## Evaluation 
### Available Models
(Take VideoChat2 as an example:)
- step 1: Download orginal models as well as weights from [VideoChat2](https://github.com/OpenGVLab/Ask-Anything/tree/main/video_chat2)
- step 2: Put choice_bench.py and open_bench.py into the folder as the same as demo.py  
- step 3: modify your path of the MLVU in choice_bench.py and open_bench.py
- step 4: run the inference and online evaluation for Multiple-choice tasks and run the inference for the generation tasks.
```
python choice_bench.py --name all
```
```
python open_bench.py --name all
```
- Step 5: run the evaluation for the generation tasks.

For Sub-Scene Captioning, modify your pred_path (by step 4) and output_dir then run
```
python evaluate_ssc.py --pred_path /VideoChat2/subplot_all.json --output_dir /eval_subplot  --output_json /eval_subplot.json
python calculate.py --path /eval_subplot
```
For Video Summarization, modify your pred_path (by step 4) and output_dir then run
```
python evaluate_summary.py --pred_path /VideoChat2/summary_all.json --output_dir /eval_summary  --output_json /eval_summary.json
```
Then run, and you need to modify the path in it to your output_dir
```
python calculate_sum.py --path /eval_summary
```


## Citation

If you find this repository useful, please consider giving a star :star: and citation

```

```

## License
MLVU is licensed under the [](). 

