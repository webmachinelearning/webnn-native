import json
import sys

def parseJson(filename, onlyFailure = False):
  failuresDict = {}
  with open(filename) as rf:
    c = json.load(rf)
    if onlyFailure:
      if c['failures'] == 0:
        return failuresDict
      else:
        testsuitesList = c['testsuites']
        for testsuite in testsuitesList:
          if testsuite['failures'] == 0:
            continue
          else:
            suiteName = testsuite['name']
            testcaseList = testsuite['testsuite']
            for testcase in testcaseList:
              if testcase.get('failures'):
                if failuresDict.get(suiteName):
                  failuresDict[suiteName].append(
                    (testcase['name'], testcase['failures']))
                else:
                  failuresDict[suiteName] = [
                    (testcase['name'], testcase['failures'])]
      return failuresDict
    else:
      return c

def getRegressionResultsList(baseline, target):
  regressionResultsList = []
  failureResultsDict = parseJson(target, True)

  if len(failureResultsDict) == 0:
    return regressionResultsList
  else:
    baseResultsDict = parseJson(baseline)
    baseSuiteList = baseResultsDict['testsuites']
    baseSuiteNameList = [
      testsuiteDict['name'] for testsuiteDict in baseSuiteList ]
    for suiteName, failureResultsList in failureResultsDict.items():
      if suiteName not in baseSuiteNameList:
        # Skip failure test case of new added testsuite
        continue
      else:
        suiteDict = baseSuiteList[baseSuiteNameList.index(suiteName)]
        baseTestcaseNameList = [
          testcaseDict['name'] for testcaseDict in suiteDict['testsuite'] ]
        for testcaseTuple in failureResultsList:
          testcaseName = testcaseTuple[0]
          if testcaseName not in baseTestcaseNameList:
            # Skip new added failure testcase
            continue
          else:
            baseTestcaseDict = \
              suiteDict['testsuite'][baseTestcaseNameList.index(testcaseName)]
            if baseTestcaseDict.get('failures') is None:
              # Catch it, this one is a regression test.
              errorMsg = '\n'.join([
                failureDict['failure'] for failureDict in testcaseTuple[1] ])
              regressionResultsList.append((suiteName, testcaseName, errorMsg))

  return regressionResultsList

if __name__ == '__main__':
  baselineFile = sys.argv[1]
  targetFile = sys.argv[2]

  resultsList = getRegressionResultsList(baselineFile, targetFile)

  if resultsList:
    print('Regression check: FAIL, %d regressoion tests:' % len(resultsList))
    for result in resultsList:
      print('[  FAILED  ] %s.%s\n%s' % (result[0], result[1], result[2]))
    sys.exit(1)
  else:
    print('Regression check: PASS')
    sys.exit(0)