from data.samples import puzzle_2
from tools.grid.utils.solver.solutionfinder import SolutionFinder


import cProfile, pstats, io
from pstats import SortKey


pr = cProfile.Profile()
pr.enable()
#TODO test 1000  samples
SolutionFinder(puzzle_2).path_to_solution()
pr.disable()



s = io.StringIO()
ps = pstats.Stats(pr, stream=s)
func_prfiles =  ps.sort_stats(SortKey.TIME).get_stats_profile().func_profiles

for funcInfo in func_prfiles.items():
    if "unblock_me" in funcInfo[1].file_name:
        print("tottime:" + str(funcInfo[1].tottime))
        print("function_name:" + funcInfo[0] )
        print("cumtime:" + str(funcInfo[1].cumtime))
        print("percall_cumtime:" + str(funcInfo[1].percall_cumtime))
        print("percall_tottime:" + str(funcInfo[1].percall_tottime))
        print("ncalls:" + str(funcInfo[1].ncalls))
   
        print("file_name:" +  str(funcInfo[1].file_name) )
        print("line_number:" +  str(funcInfo[1].line_number) )
        print("-----------------------------------------------------")


