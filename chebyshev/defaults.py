from .refinement import ErrorControl,IntervalNumberControl,GridRegularityControl,RefinementScheme




min_deg = 4
max_deg = 8
max_abs_err = 1e-2
max_int_count = 2**9
max_ref_count_per_cycle = 2**4
grid_condition_bound = 50
max_num_cycles_per_ref = 2**4

error_control = ErrorControl(min_deg,max_deg,max_abs_err=max_abs_err)
interval_number_control = IntervalNumberControl(max_ref_count_per_cycle,max_int_count)
grid_regularity_control = GridRegularityControl(grid_condition_bound)
refinement_scheme = RefinementScheme(error_control,\
                interval_number_control,\
                grid_regularity_control,max_num_cycles=max_num_cycles_per_ref)
