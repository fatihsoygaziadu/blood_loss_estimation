# -*- coding: utf-8 -*-
import logging
import os
from Algorithms import ImageConfig


def log_print(*msg_list):

    msg_str_list = ''
    for m_str in msg_list:
        msg_str_list += str(m_str)

    # logger = logging.getLogger("crumbs")
    logger = logging.getLogger('Image_Logger')
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s \t [%(levelname)s | %(filename)s:%(lineno)s] > %(message)s')

    dirname = "./log"
    # dirname = ImageConfig.root_folder

    if not os.path.isdir(dirname):
        os.mkdir(dirname)
        # fileHandler = logging.FileHandler(dirname + "/log_" + ImageConfig.folder_data_type + '_' + ImageConfig.ts now.strftime("%Y-%m-%d")+".log")
    fileHandler = logging.FileHandler(dirname + "/log_" + ImageConfig.folder_data_type + '_' + ImageConfig.ts + ".log")

    streamHandler = logging.StreamHandler()

    fileHandler.setFormatter(formatter)
    streamHandler.setFormatter(formatter)

    logger.addHandler(fileHandler)
    logger.addHandler(streamHandler)

    logger.info(msg_str_list)

    logger.removeHandler(streamHandler)
    logger.removeHandler(fileHandler)