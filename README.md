# MLVU: Multi-task Long Video Understanding Benchmark

[Paper](Arxiv Link) [Dataset](https://huggingface.co/datasets/JUNJIE99/MLVU)

## License
Our dataset is under the CC-BY-NC-SA-4.0 license. We do not own the copyright of any original video files. If you need to access and use our dataset, you must understand and agree: this dataset is for research purposes only and cannot be used for any commercial or other purposes. The user assumes all effects arising from any other use and dissemination.



## News 
- 6/5/2024: Release :fire:


## Hosting and Maintenance
The annotation files will be permanently retained. At this stage, we provide video access to users under the condition of acknowledging the above license. For the video data used, we respect and acknowledge any copyrights of the video authors. Therefore, for the movies, TV shows, documentaries, and cartoons used in the dataset, we have reduced the resolution, clipped the length, adjusted dimensions, etc. of the original videos to minimize the impact on the rights of the original works. If the original authors of the related works still believe that the videos should be removed, please contact mlvubenchmark@gmail.com or directly raise an issue.

If some videos are requested to be removed, we will replace them with a set of video frames sparsely sampled from the video and adjusted in resolution. Since all the questions in MLVU are only related to visual content and do not involve audio, this will not significantly affect the validity of MLVU (most existing MLLMs also understand videos by frame extraction).

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

