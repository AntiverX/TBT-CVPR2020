import logging
import os

logging.basicConfig(filename=os.path.join("log.log"),
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.info("test logger.")