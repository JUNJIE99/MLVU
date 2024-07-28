## Evaluation for MLVU Test

We provide detailed evaluation methods for MLVU Test, including Multiple-choice tasks and Generation tasks.

#### Multiple-Choice testing
- Step 1 Get the inference results by running the following scripts (Take Video-LLaVA as an example).
```
python test_bench.py 
```
You can get all the outputs in one file in the following format like [test_res.json](https://github.com/JUNJIE99/MLVU/blob/main/evaluation_test/test_res.json):
```
 {
        "question_id": "AR_0",
        "question_type": "anomaly_reco",
        "option": "B"
}
...
```
- Step 2 Submit your json files to the MLVU online evaluation system to get the final results.
  
#### Generation testing
Please see the detail steps provided in the [eval folder](https://github.com/JUNJIE99/MLVU/tree/main/evaluation).
