import os
import sys
import time
import logging
from datetime import timedelta


start = time.time()
logging.basicConfig(filename='./timer.txt', format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

benchlist = [sys.argv[1]]
TIMEOUT = int(sys.argv[2])

for B in benchlist:
    logging.info(f"Bench --> {B}")
    for epsilon in [4]:
        _start = time.time()

        os.system(f"cd {B}/run; \
                    python3 generate-include.py; \
                    python3 setup.py {B}; \
                    rm -f *.txt; \
                    python3 create-search-space.py {B}; \
                    python3 ../dd2.py {B} search_config.json config.json {TIMEOUT} {epsilon} A \
                        ")
        _elapsed = (time.time() - _start)
        logging.info(f"      epsilon --> {epsilon}  time --> {str(timedelta(seconds=_elapsed))}")

elapsed = (time.time() - start)
logging.info(f"TOTAL TIME: {str(timedelta(seconds=elapsed))}")