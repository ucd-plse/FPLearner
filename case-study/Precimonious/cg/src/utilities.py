import logging
import os

seg_time = 9999
HIGHEST =-1
LOWEST = 0


def get_digit(str_name):
  digit = ""
  for s in str_name:
    if s.isdigit():
      digit += s
  return int(digit)


def parse_json(search_config):
  search_config_len = len(search_config)
  zero_one_vec = [-1] * search_config_len
  for item in search_config:
    idx = get_digit(item) - 1
    zero_one_vec[idx]= 1 if "double" in str(search_config[item]["type"]) else 0

  return zero_one_vec


def get_dynamic_score():
  score = seg_time
  if os.path.isfile("time.txt"):
    with open("time.txt") as scorefile:
      temp = scorefile.readline().strip()
      if temp:
        score = temp
  return float(score)


def check_error():
  if os.path.isfile("log.txt"):
    with open("log.txt") as f:
      firstline = f.readline().rstrip()

     # FOR CG:
      zeta = f.readline().rstrip()
      err = f.readline().rstrip()
      logging.info("within err threshold: {}, zeta = {}, error = {}".format(firstline, zeta, err))

     # FOR EP:
      # sx = f.readline().rstrip()
      # sx_err = f.readline().rstrip()
      # sy = f.readline().rstrip()
      # sy_err = f.readline().rstrip()
      # logging.info("within err threshold: {}, sx = {}, sx_err = {}, sy = {}, sy_err = {}".format(firstline, sx, sx_err, sy, sy_err))

     # For BT:
      # logging.info("within err threshold: {}".format(firstline))


    if firstline == "true":
      return 1
    else:
      return 0
  else:
#     segmentation fault
    logging.info("segmentation fault")
    return -1


def is_empty(type_set):
  for t in type_set:
    if len(t) > 1:
      return False
  return True


#
# modify change set so that each variable
# maps to its highest type
#
def to_highest_precision(change_set, type_set, switch_set):
  for i in range(0, len(change_set)):
    c = change_set[i]
    t = type_set[i]
    if len(t) > 0:
      c["type"] = t[HIGHEST]
    if len(switch_set) > 0:
      s = switch_set[i]
      if len(s) > 0:
        c["switch"] = s[HIGHEST]


#
# modify change set so that each variable
# maps to its 2nd highest type
#
def to_2nd_highest_precision(change_set, type_set, switch_set):
  for i in range(0, len(change_set)):
    c = change_set[i]
    t = type_set[i]
    if len(t) > 1:
      c["type"] = t[LOWEST]
    if len(switch_set) > 0:
      s = switch_set[i]
      if len(s) > 1:
        c["switch"] = s[LOWEST]

