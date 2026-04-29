# import logging, os

# def setup_logger(log_file):
#     os.makedirs(os.path.dirname(log_file), exist_ok=True)
#     logger = logging.getLogger("UAV_Tracker")
#     logger.setLevel(logging.INFO)
#     fh = logging.FileHandler(log_file)
#     ch = logging.StreamHandler()
#     fmt = logging.Formatter(
#         "%(asctime)s [%(levelname)s] %(message)s",
#         datefmt="%Y-%m-%d %H:%M:%S"
#     )
#     fh.setFormatter(fmt)
#     ch.setFormatter(fmt)
#     logger.addHandler(fh)
#     logger.addHandler(ch)
#     return logger

# def log_frame(logger, frame_idx, fps, inf_time, num_tracks, id_switches):
#     logger.info(
#         f"Frame {frame_idx:05d} | FPS:{fps:.2f} | "
#         f"InfTime:{inf_time:.2f}ms | Tracks:{num_tracks} | IDsw:{id_switches}"
#     )

import logging
import os

def setup_logger(log_file):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger("UAV_Tracker")
    logger.setLevel(logging.INFO)
    
    fh = logging.FileHandler(log_file)
    ch = logging.StreamHandler()
    
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

def log_frame(logger, frame_idx, fps, inf_time, num_tracks, id_switches):
    logger.info(
        f"Frame {frame_idx:05d} | FPS:{fps:.2f} | "
        f"InfTime:{inf_time:.2f}ms | Tracks:{num_tracks} | IDsw:{id_switches}"
    )