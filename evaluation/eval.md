## Evaluation for MLVU

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
