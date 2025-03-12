import os
import sys
import logging

class Logger(logging.Logger):

    def __init__(self, name, level=logging.NOTSET, output="filestream", fileName='logger.log',filePath=os.curdir,
                 formatting="%(asctime)s - %(levelname)s - %(message)s"):
        
        super().__init__(name, level)
        self.extra_info = None

        logg_dir = os.path.join(filePath, 'logs')
        if not os.path.exists(logg_dir):
            os.mkdir(logg_dir)

        if output == "filestream" or output == "all":
            handler = logging.FileHandler(filename=os.path.join(logg_dir,fileName))            
            handler.setFormatter(logging.Formatter(formatting))
            self.addHandler(handler)
        
        if output == 'stdout' or output == "all":
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter(formatting))
            self.addHandler(handler)


    def createLogMessage(self, prefix, message):
        return "[" + prefix + "] " + message

    def debug(self, prefix, message, *args, **kwargs):
        super().debug(self.createLogMessage(prefix,message), *args, **kwargs)

    def info(self, prefix, message, *args, **kwargs):
        super().info(self.createLogMessage(prefix,message), *args, **kwargs)

    def warn(self, prefix, message, *args, **kwargs):
        super().warn(self.createLogMessage(prefix,message), *args, **kwargs)

    def error(self, prefix, message, *args, **kwargs):
        super().error(self.createLogMessage(prefix,message), *args, **kwargs)

    def fatal(self, prefix, message, *args, **kwargs):
        super().critical(self.createLogMessage(prefix,message), *args, **kwargs)