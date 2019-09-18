import os
import foundations

NUM_JOBS = 10

for job_num in range(NUM_JOBS):
    print(f'job number {job_num}')
    foundations.submit(
        scheduler_config="scheduler",
        command=["main.py"],
        project_name="Trump_Twitter_ML_Experiments"
            )