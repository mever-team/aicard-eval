from aicard_eval.emissions import CarbonTrack

# initialize the tracker
emission_tracker = CarbonTrack()
# initialize run
run_name = 'my_run'
emission_tracker.start(run_name)
# execute some code
#some_code()
# stop the tracker and obtain the output
emissions = emission_tracker.stop(run_name)
emissions_out = {k: emissions[k] for k in ['energy_consumed', 'emissions', 'cpu_model', 'gpu_model', 'ram_total_size']}
print(emissions_out)
