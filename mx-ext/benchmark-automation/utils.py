import logging
import psutil
import os

def configure_logger(prefix: str = "",log_file = None):
    format = f"[%(asctime)s{prefix}] %(message)s"
    # format = f"[%(asctime)s.%(msecs)03d{prefix}] %(message)s"
    logging.basicConfig(
        level=logging.DEBUG,
        format=format,
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
        filename = log_file
    )


def kill_all(kill_list):
    current_pid = os.getpid()
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        if not proc.info['cmdline']:
            continue
        try:
            cmdline = ' '.join(proc.info['cmdline'])
            print(cmdline)
            pass
            for kill in kill_list:
                if kill in cmdline and proc.pid != current_pid:
                    try:
                        proc.kill()   
                        print(f"has killed {proc.pid}")
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue