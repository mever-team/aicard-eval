from typing import Any
import threading
import time
import json
from codecarbon import EmissionsTracker


class Job():
    def __init__(self, 
                 worker: str|None = None, 
                 operation: str|None = None, 
                 data: Any = None,
                 carbon_tracker: EmissionsTracker|None = None):
        self.worker = worker
        self.operation = operation
        self.data = data
        self.start = None
        self.finish = None
        self.carbon_tracker = carbon_tracker
        
    def __repr__(self):
        return f"worker={self.worker}, operation={self.operation}, data={self.data}"
        
    def to_dict(self):
        return {
            "worker": self.worker,
            "operation": self.operation,
            "data": self.data,
        }
        
    def __str__(self):
        return f"worker={self.worker}, operation={self.operation}, data={self.data}"

class CardJobsTracker():
    __jobs = {} # active
    __last_job = {} # last finished job of each card_id
    __last_id = 0
    __to_flash_ids = []
    
    def __init__(self, logger = None):
        self.logger = logger
        gc_thread = threading.Thread(target=self.gc, daemon=True)
        gc_thread.start()
    
    def set(self, name: Any, job: Job):
        if name in self.__jobs.keys():
            if self.logger: 
                self.logger.error(f'There is already a job with the name {name}')
            else: 
                print(f'There is already a job with the name {name}')
            return
        self.__last_id += 1
        job_id = self.__last_id
        job.id = job_id
        job.start = time.time()
        self.__jobs[name] = job
        return True
    
    def update(self, name: Any, job: Job):
        if name not in self.__jobs.keys():
            if self.logger: 
                self.logger.error(f'There is no job with the name {name}')
            else:
                print(f'There is no job with the name {name}')
            return
        old_job = self.__jobs[name]
        old_job_id = old_job.id
        job.id = old_job_id
        self.__jobs[name] = job
        return True
    
    def get(self, name: Any):
        return self.__jobs.get(name, None)
    
    def get_last(self, name: Any):
        return self.__last_job.get(name, None)
    
    def set_last(self, name: Any, job: Job):
        self.__last_job[name] = job
    
    def pop_last(self, name: Any):
        return self.__last_job.pop(name, None)

    def delete(self, name: Any, data: Any = None):
        if name not in self.__jobs:
            if self.logger: 
                self.logger.warn(f'No job for card {name} to remove')
            else:
                print(f'No job for card {name} to remove')
            return False
        job = self.get(name)
        if data: job.data = data # optionally replace data at final state
        job.finish = time.time()
        self.set_last(name, job)
        self.__to_flash_ids.append(name)
        self.__jobs.pop(name)
        return True
    
    def gc(self):
        while True:
            now = time.time()
            for name, job in self.__jobs.items():
                finished_at = job.finish
                dt = now - finished_at
                if dt > 600:
                    self.__jobs.pop(name)
            for name, job in self.__last_job.items():
                finished_at = job.finish
                dt = now - finished_at
                if dt > 1*24*60*60: # keep self.__last_job for 1 days
                    self.__last_job.pop(name)
            time.sleep(600) # keep self.__jobs for 10 minutes
            
    def interval_overlap(self, job1, job2):
        now = time.time()
        finish1 = job1.finish if job1.finish is not None else now
        finish2 = job2.finish if job2.finish is not None else now

        latest_start = max(job1.start, job2.start)
        earliest_finish = min(finish1, finish2)

        overlap = earliest_finish - latest_start
        return max(overlap, 0.0) # overlap duration (in seconds) between two jobs
    
    def flash_codecarbon(self, job_name):
        if job_name not in self.__to_flash_ids:
            return {}
        emissions =  self.estimate_codecarbon(job_name)
        self.__to_flash_ids.remove(job_name)
        return emissions
        
    def estimate_codecarbon(self, job_name):
        job = self.get_last(job_name)
        if not job:
            job_active = self.get(job_name)
            # use this print for debug only because this blockes is used during client polling
            # if not job_active:
            #     if self.logger: 
            #         self.logger.error(f"Job {job_name} dons't exist")
            #     else:
            #         print(f"Job {job_name} dons't exist")
            # else:
            #     if self.logger: 
            #         self.logger.error(f"Job {job_name} hasn't finished")
            #     else:
            #         print(f"Job {job_name} hasn't finished")
                    
            return {}
        energy_consumed = job.data.get('energy_consumed', None)
        emissions = job.data.get('emissions', None)
        execution_time = job.finish - job.start
        if not energy_consumed or not emissions: 
            if self.logger: 
                self.logger.error(f"No code carbon data found in the job {job_name}")
            else:
                print(f"No code carbon data found in the job {job_name}")
                
            return {}
        # Caluclate percentage worload
        all_jobs = self.__jobs | self.__last_job
        time_overlay = 0
        for other_name, other_job in all_jobs.items():
            if other_name == job_name:
                continue
            time_overlay += self.interval_overlap(job, other_job)
        cumulated_time = execution_time + time_overlay
        percent_workload = execution_time / cumulated_time
        job.data.update({"emissions": emissions*percent_workload, "energy_consumed": energy_consumed*percent_workload})
        return job.data
    
    
class CarbonTrack():
    def __init__(self):
        self.__job_tracker = CardJobsTracker()
        
    def start(self, name):
        carbon_tracker = EmissionsTracker( project_name=name, save_to_file=False, log_level="WARNING", tracking_mode="process")
        job = Job(carbon_tracker=carbon_tracker)
        self.__job_tracker.set(name, job)
        job.carbon_tracker.start()
        
    def stop(self, name):
        job = self.__job_tracker.get(name)
        job.carbon_tracker.stop()
        emissions_data = json.loads(job.carbon_tracker.final_emissions_data.toJSON())
        self.__job_tracker.delete(name, emissions_data)
        return self.__job_tracker.estimate_codecarbon(name)