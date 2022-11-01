import os
import psutil
import time


def find_procs_by_name(name):
    "Return a list of processes matching 'name'."
    assert name, name
    ls = []
    for p in psutil.process_iter():
        name_, exe, cmdline = "", "", []
        try:
            name_ = p.name()
            cmdline = p.cmdline()
            exe = p.exe()
        except (psutil.AccessDenied, psutil.ZombieProcess):
            pass
        except psutil.NoSuchProcess:
            continue
        if len(cmdline) < 1:
            continue
        if name == name_ or cmdline[0] == name or os.path.basename(exe) == name:
            ls.append(p)
    return ls


if __name__ == "__main__":
    process = find_procs_by_name("MultiSense-viewer")[0]
    f = open("ResLog.csv", "w")
    f.write("cpu, mem, net in, net out, \n")

    # get the network I/O stats from psutil
    # extract the total bytes sent and received
    seconds = 5
    minutes = 0
    t_end = time.time() + (5) + seconds
    log = 0
    while time.time() < t_end:
        cpu = process.cpu_percent()
        mem = process.memory_info().rss / 1000000
        io = psutil.net_io_counters()
        bytes_sent, bytes_recv = io.bytes_sent, io.bytes_recv

        if log < 5:
            cpu = process.cpu_percent()
            log += 1
            continue
        print("{}: {},{},{},{}\n".format(log, cpu, mem, bytes_recv, bytes_sent))

        f.write("{},{},{},{},\n".format(cpu, mem, bytes_recv, bytes_sent))
        time.sleep(1)
        log += 1

    f.close()
