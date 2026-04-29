# import sys, os
# sys.path.insert(0, os.path.dirname(__file__))

# import config
# from logger import setup_logger, log_frame
# from inference import run_inference

# def main():
#     logger = setup_logger(config.LOG_FILE)
#     logger.info("="*50)
#     logger.info("UAV Maritime MA-MOT System Starting")
#     logger.info(f"Model:  {config.MODEL_PATH}")
#     logger.info(f"Input:  {config.INPUT_PATH}")
#     logger.info(f"Meta:   {config.METADATA_PATH}")
#     logger.info("="*50)

#     if not os.path.exists(config.MODEL_PATH):
#         logger.error(f"Model not found: {config.MODEL_PATH}")
#         sys.exit(1)

#     if not os.path.exists(config.INPUT_PATH):
#         logger.error(f"Input not found: {config.INPUT_PATH}")
#         sys.exit(1)

#     logger.info("Starting MA-MOT inference pipeline...")

#     inf_time, avg_fps, id_sw = run_inference(
#         video_path  = config.INPUT_PATH,
#         model_path  = config.MODEL_PATH,
#         meta_path   = config.METADATA_PATH if config.USE_METADATA else None,
#         conf        = config.CONF_THRESHOLD,
#         output_path = config.OUTPUT_VIDEO
#     )

#     logger.info("FINAL RESULTS:")
#     logger.info(f"  Avg Inference Time : {inf_time:.2f} ms")
#     logger.info(f"  Average FPS        : {avg_fps:.2f}")
#     logger.info(f"  Total ID Switches  : {id_sw}")
#     logger.info(f"  Output saved to    : {config.OUTPUT_VIDEO}")
#     logger.info("System shutdown complete.")

# if __name__ == "__main__":
#     main()

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from logger import setup_logger
from inference import run_inference


def main():
    logger = setup_logger(config.LOG_FILE)
    logger.info("=" * 50)
    logger.info("UAV Maritime MA-MOT System Starting")
    logger.info(f"Model:  {config.MODEL_PATH}")
    logger.info(f"Input:  {config.INPUT_PATH}")
    logger.info(f"Meta:   {config.METADATA_PATH}")
    logger.info("=" * 50)

    if not os.path.exists(config.MODEL_PATH):
        logger.error(f"Model not found: {config.MODEL_PATH}")
        sys.exit(1)

    if not os.path.exists(config.INPUT_PATH):
        logger.error(f"Input not found: {config.INPUT_PATH}")
        sys.exit(1)

    logger.info("Starting MA-MOT inference pipeline...")

    inf_time, avg_fps, id_sw = run_inference(
        video_path=config.INPUT_PATH,
        model_path=config.MODEL_PATH,
        meta_path=config.METADATA_PATH if config.USE_METADATA else None,
        conf=config.CONF_THRESHOLD,
        output_path=config.OUTPUT_VIDEO
    )

    logger.info("FINAL RESULTS:")
    logger.info(f"  Avg Inference Time : {inf_time:.2f} ms")
    logger.info(f"  Average FPS        : {avg_fps:.2f}")
    logger.info(f"  Total ID Switches  : {id_sw}")
    logger.info(f"  Output saved to    : {config.OUTPUT_VIDEO}")
    logger.info("System shutdown complete.")


if __name__ == "__main__":
    main()