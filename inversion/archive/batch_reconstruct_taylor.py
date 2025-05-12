import os
import multiprocessing

def run_subprocess(year, gpu):
    subprocess_cmd = f"python reconstruct_taylor.py --year={year} --gpu={gpu}"
    os.system(subprocess_cmd)

if __name__ == "__main__":
    years = [2007, 2011, 2015, 2017, 2023]
    processes = []
    for i, year in enumerate(years):
        p = multiprocessing.Process(target=run_subprocess, args=(year, i,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
