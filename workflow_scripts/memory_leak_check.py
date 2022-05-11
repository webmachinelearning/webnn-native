import sys
from os.path import exists

def memoryLeakCheck(resultFile):
  with open(resultFile) as rf:
    lines = rf.readlines()
    for line in lines:
      if line.find('Detected memory leaks!') != -1:
        raise Exception('Detected memory leaks!')

if __name__ == '__main__':
  resultFile = sys.argv[1]

  if not exists(resultFile):
      print("Failed to get result file: %s, please have a check" % \
            resultFile)
      sys.exit(1)

  memoryLeakCheck(resultFile)