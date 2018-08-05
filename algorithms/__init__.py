from abc import ABC, abstractmethod
import logging
import time

class Algorithm(ABC):

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def run(self):
        start_time = time.time()

        self._run()

        end_time = time.time()
        self.logger.info(f'Took {int(end_time - start_time)} seconds to converge.')
    
    @abstractmethod
    def _run(self):
        pass