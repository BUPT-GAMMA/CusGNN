import logging
import time
import logging.handlers as handlers
import sys
class SizedTimedRotatingFileHandler(handlers.TimedRotatingFileHandler):
    """
    Handler for logging to a set of files, which switches from one file
    to the next when the current file reaches a certain size, or at certain
    timed intervals
    """
    def __init__(self, filename, mode='a', maxBytes=0, backupCount=0, encoding=None,
                 delay=0, when='h', interval=1, utc=False):
        # If rotation/rollover is wanted, it doesn't make sense to use another
        # mode. If for example 'w' were specified, then if there were multiple
        # runs of the calling application, the logs from previous runs would be
        # lost if the 'w' is respected, because the log file would be truncated
        # on each run.
        if maxBytes > 0:
            mode = 'a'
        handlers.TimedRotatingFileHandler.__init__(
            self, filename, when, interval, backupCount, encoding, delay, utc)
        self.maxBytes = maxBytes

    def shouldRollover(self, record):
        """
        Determine if rollover should occur.

        Basically, see if the supplied record would cause the file to exceed
        the size limit we have.
        """
        if self.stream is None:                 # delay was set...
            self.stream = self._open()
        if self.maxBytes > 0:                   # are we rolling over?
            msg = "%s\n" % self.format(record)
            self.stream.seek(0, 2)  #due to non-posix-compliant Windows feature
            if self.stream.tell() + len(msg) >= self.maxBytes:
                return 1
        t = int(time.time())
        if t >= self.rolloverAt:
            return 1
        return 0

def init_logger(logger_name='', log_file='', log_level='', print_console=False):

    logger = logging.getLogger()
    logger.setLevel(log_level)

    # create a file handler
    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.INFO)


    error_handler = logging.StreamHandler(sys.stdout)
    error_handler.setLevel(logging.ERROR)

    # create a logging format
    formatter = logging.Formatter('%(name)s-logging.%(levelname)s-%(thread)d-%(asctime)s-%(message)s')
    handler.setFormatter(formatter)
    error_handler.setFormatter(formatter)

    daily_handler=SizedTimedRotatingFileHandler(log_file, when='midnight')
    daily_handler.setLevel(logging.INFO)
    daily_handler.setFormatter(formatter)

    # add the handlers to the logger
    #logger.addHandler(handler)
    logger.addHandler(error_handler)
    logger.addHandler(daily_handler)

    if print_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    return logger