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
- 🏠 6/19/2024: For better maintenance and updates of MLVU, we have migrated MLVU to this new repository. We will continue to update and maintain MLVU here. If you have any questions, feel free to raise an issue. :fire:

## License
Our dataset is under the CC-BY-NC-SA-4.0 license.

:warning: If you need to access and use our dataset, you must understand and agree: **This dataset is for research purposes only and cannot be used for any commercial or other purposes. The user assumes all effects arising from any other use and dissemination.**

We do not own the copyright of any raw video files. Currently, we provide video access to researchers under the condition of acknowledging the above license. For the video data used, we respect and acknowledge any copyrights of the video authors. Therefore, for the movies, TV series, documentaries, and cartoons used in the dataset, we have reduced the resolution, clipped the length, adjusted dimensions, etc. of the original videos to minimize the impact on the rights of the original works. 

If the original authors of the related works still believe that the videos should be removed, please contact mlvubenchmark@gmail.com or directly raise an issue.


## Introduction
We introduce MLVU: the first comprehensive benchmark designed for evaluating Multimodal Large Language Models (MLLMs) in Long Video Understanding (LVU) tasks. MLVU is constructed from a wide variety of long videos, with lengths ranging from 3 minutes to 2 hours, and includes nine distinct evaluation tasks. These tasks challenge MLLMs to handle different types of tasks, leveraging both global and local information from videos. 

Our evaluation of 20 popular MLLMs, including GPT-4o, reveals significant challenges in LVU, with even the top-performing GPT-4o only achieving an average score of 64.6% in multi-choice tasks. In addition, our empirical results underscore the need for improvements in context length, image understanding, and strong LLM-backbones. We anticipate that MLVU will serve as a catalyst for the community to further advance MLLMs' capabilities in understanding long videos.

![Statistical overview of our LVBench dataset. **Left:** Distribution of Video Duration; **Middle** Distribution of Source Types for Long Videos; **Right:** Quantification of Each Task Type.](./figs/statistic.png)



## :trophy: Mini-Leaderboard

| Model | Input | M-Avg | G-Avg |
| --- | --- | --- | --- |
| Full mark | - | 100 | 10 |
| [GPT-4o](https://openai.com/index/hello-gpt-4o/) | 0.5 fps | 64.6 | 5.80 |
| [LongVA](https://github.com/EvolvingLMMs-Lab/LongVA) | 256 frm | 56.3 | 4.33 |
| [InternVL-1.5](https://github.com/OpenGVLab/InternVL) | 16 frm | 50.4 | 4.02 |
| [GPT-4 Turbo](https://openai.com/index/gpt-4v-system-card/) | 16 frm | 49.2 | 5.35 |
| [VideoLLaMA2-Chat](https://github.com/DAMO-NLP-SG/VideoLLaMA2) | 16 frm | 48.5 | 3.99 |
| [VideoChat2_HD](https://github.com/OpenGVLab/Ask-Anything/tree/main/video_chat2) | 16 frm | 47.9 | 3.99 |
| [Video-LLaVA](https://github.com/PKU-YuanGroup/Video-LLaVA) | 8 frm | 47.3 | 3.84 |
| [ShareGPT4Video](https://github.com/ShareGPT4Omni/ShareGPT4Video) | 16 frm | 46.4 | 3.77 |
| [VideoChat2-Vicuna](https://github.com/OpenGVLab/Ask-Anything/tree/main/video_chat2) | 16 frm | 44.5 | 3.81 |
| [MiniGPT4-Video](https://github.com/Vision-CAIR/MiniGPT4-video) | 90 frm | 44.5 | 3.36 |
| [Qwen-VL-Max](https://github.com/QwenLM/Qwen) | 16 frm | 42.2 | 3.96 |
| [LLaVA-1.6](https://github.com/haotian-liu/LLaVA) | 16 frm | 39.3 | 3.23 |
| [Claude-3-Opus](https://claude.ai/login?returnTo=%2F%3F) | 16 frm | 36.5 | 3.39 |
| [MA-LMM](https://github.com/boheumd/MA-LMM) | 1000 frm | 36.4 | 3.46 |
| [Video-LLaMA-2](https://github.com/DAMO-NLP-SG/Video-LLaMA) | 16 frm | 35.5 | 3.78 |
| [LLaMA-VID](https://github.com/dvlab-research/LLaMA-VID) | 1 fps | 33.2 | 4.22 |
| [Video-ChatGPT](https://github.com/mbzuai-oryx/Video-ChatGPT) | 100 frm | 31.3 | 3.90 |
| [TimeChat](https://github.com/RenShuhuai-Andy/TimeChat) | 96 frm | 30.9 | 3.42 |
| [VideoChat](https://github.com/OpenGVLab/Ask-Anything/tree/main/video_chat) | 16 frm | 29.2 | 3.66 |
| [Movie-LLM](https://github.com/Deaddawn/MovieLLM-code) | 1 fps | 26.1 | 3.94 |
| [mPLUG-Owl-V](https://github.com/X-PLUG/mPLUG-Owl) | 16 frm | 25.9 | 3.84 |
| [MovieChat](https://github.com/rese1f/MovieChat) | 2048 frm | 25.8 | 2.78 |
| [Otter-V](https://github.com/Luodian/Otter) | 16 frm | 24.4 | 3.31 |
| [Otter-I](https://github.com/Luodian/Otter) | 16 frm | 23.3 | 3.15 |







## License
Our dataset is under the CC-BY-NC-SA-4.0 license.

:warning: If you need to access and use our dataset, you must understand and agree: **This dataset is for research purposes only and cannot be used for any commercial or other purposes. The user assumes all effects arising from any other use and dissemination.**

We do not own the copyright of any raw video files. Currently, we provide video access to researchers under the condition of acknowledging the above license. For the video data used, we respect and acknowledge any copyrights of the video authors. Therefore, for the movies, TV series, documentaries, and cartoons used in the dataset, we have reduced the resolution, clipped the length, adjusted dimensions, etc. of the original videos to minimize the impact on the rights of the original works. 

If the original authors of the related works still believe that the videos should be removed, please contact mlvubenchmark@gmail.com or directly raise an issue.

## MLVU Benchmark
> Before you access our dataset, we kindly ask you to thoroughly read and understand the license outlined above. If you cannot agree to these terms, we request that you refrain from downloading our video data.


The annotation file is readily accessible [here](https://github.com/FlagOpen/FlagEmbedding/tree/master/MLVU/data). For the raw videos, you can access them via this [<u>🤗 HF Link</u>](https://huggingface.co/datasets/MLVU/MVLU).


MLVU encompasses nine distinct tasks, which include multiple-choice tasks as well as free-form generation tasks. These tasks are specifically tailored for long-form video understanding, and are classified into three categories: holistic understanding, single detail understanding, and multi-detail understanding. Examples of the tasks are displayed below.


![Task Examples of our MLVU.](./figs/task_example.png)


## Evaluation
Please refer to our [evaluation](https://github.com/FlagOpen/FlagEmbedding/tree/master/MLVU/evaluation) folder for more details.




## Hosting and Maintenance
The annotation files will be permanently retained. 

If some videos are requested to be removed, we will replace them with a set of video frames sparsely sampled from the video and adjusted in resolution. Since **all the questions in MLVU are only related to visual content** and do not involve audio, this will not significantly affect the validity of MLVU (most existing MLLMs also understand videos by frame extraction).

If even retaining the frame set is not allowed, we will still keep the relevant annotation files, and replace them with the meta-information of the video, or actively seek more reliable and risk-free video sources.





## Citation

If you find this repository useful, please consider giving a star :star: and citation

```
@article{MLVU,
  title={MLVU: A Comprehensive Benchmark for Multi-Task Long Video Understanding},
  author={Zhou, Junjie and Shu, Yan and Zhao, Bo and Wu, Boya and Xiao, Shitao and Yang, Xi and Xiong, Yongping and Zhang, Bo and Huang, Tiejun and Liu, Zheng},
  journal={arXiv preprint arXiv:2406.04264},
  year={2024}
}
```


