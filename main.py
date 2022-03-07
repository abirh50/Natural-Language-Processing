# Abir Haque
# CSCI 381 NLP
# HW 1


import math

def main():
    # Unigram maximum likelihood model
    uniTrainingWords = {}
    uniTrainingWordsWithoutUnk = {} # has no <unk>
    uniCountWords("train.txt", uniTrainingWords, uniTrainingWordsWithoutUnk)

    uniTestWords = {}
    uniTestWordsWithoutUnk = {} # has no <unk>
    uniCountWords("test.txt", uniTestWords, uniTestWordsWithoutUnk)

    addStartEndUnkTag("test.txt", "train.txt", uniTrainingWords, uniTrainingWordsWithoutUnk, uniTestWords, uniTestWordsWithoutUnk)

    # Bigram maximum likelihood model
    biTrainingWords = {}
    biTestWords = {}
    biCountWords("processedTrain.txt", biTrainingWords)
    biCountWords("processedTest.txt", biTestWords)

    # Question 1
    print("Question 1:")
    print("Word types:", len(uniTrainingWords) - 1)
    print("----------------------------------------------")
    print()

    # Question 2
    print("Question 2:")
    print("Word tokens:", trainingWordTokens(uniTrainingWords) - uniTrainingWords["<s>"])
    print("----------------------------------------------")
    print()

    # Question 3
    print("Question 3:")
    print("% of Word tokens in test corpus not in training corpus:", "{}%".format(testTokensNotInTrain_Percent(uniTestWordsWithoutUnk, uniTrainingWordsWithoutUnk, "unigram", 0)))
    print("% of Word types in test corpus not in training corpus:", "{}%".format(testTypesNotInTrain_Percent(uniTestWordsWithoutUnk, uniTrainingWordsWithoutUnk, "unigram")))
    print("----------------------------------------------")
    print()

    # Question 4
    print("Question 4:")
    print("% of bigram tokens in test corpus not in training corpus:", "{}%".format(testTokensNotInTrain_Percent(biTestWords, biTrainingWords, "bigram", uniTestWords["<s>"])))
    print("% of bigram types in test corpus not in training corpus:", "{}%".format(testTypesNotInTrain_Percent(biTestWords, biTrainingWords, "bigram")))
    print("----------------------------------------------")
    print()

    # Question 5
    str = "I look forward to hearing your reply ."
    str = str.lower()
    strDict = {}
    perplexity = {}
    print("Question 5:")
    print("Log probability of unigram model:")
    logProb(str, strDict, uniTrainingWords, "unigram", perplexity)
    print("Log probability of bigram model:")
    logProb(str, uniTrainingWords, biTrainingWords, "bigram", perplexity)
    print("Log probability of bigram model with Add-One smoothing:")
    logProb(str, uniTrainingWords, biTrainingWords, "bao", perplexity)
    print("----------------------------------------------")
    print()

    # Question 6
    print("Question 6:")
    print("Unigram model perplexity:", perplexity["unigram"])
    print("Bigram model perplexity:", perplexity["bigram"])
    print("Bigram model with Add-One smoothing perplexity:", perplexity["bao"])
    print("----------------------------------------------")
    print()

    # Question 7
    testTokens = wordTokens(uniTestWords)
    print("Question 7:")
    tLogProb("processedTest.txt", uniTestWords, uniTrainingWords, "tUnigram", perplexity, testTokens)
    tLogProb("processedTest.txt", uniTrainingWords, biTrainingWords, "tBigram", perplexity, testTokens)
    tLogProb("processedTest.txt", uniTrainingWords, biTrainingWords, "tBAO", perplexity, testTokens)

    print("Unigram model perplexity:", perplexity["tUnigram"])
    print("Bigram model perplexity:", perplexity["tBigram"])
    print("Bigram model with Add-One smoothing perplexity:", perplexity["tBAO"])

# log probability for processedTest.txt
def tLogProb(testFile, dict, trainDict, type, perplexity, testTokens):
    File = open(testFile, "r+")
    sum = 0
    undef = False
    if type == "tUnigram":
        for line in File:
            for word in line.split():
                if word in trainDict:
                    logProba = math.log2(trainDict[word] / wordTokens(trainDict))
                    sum += logProba
                else:
                    undef = True
                    break
            if undef:
                break
    elif type == "tBigram":
        for line in File:
            wordList = line.split()
            for i in range(len(wordList) - 1):
                biWords = (wordList[i], wordList[i + 1])
                if biWords in trainDict:
                    logProba = math.log2(trainDict[biWords] / dict[wordList[i]])
                else:
                    undef = True
                    break
                sum += logProba
            if undef:
                break
    else:
        for line in File:
            wordList = line.split()
            for i in range(len(wordList) - 1):
                biWords = (wordList[i], wordList[i + 1])
                if biWords in trainDict:
                    logProba = math.log2((trainDict[biWords] + 1) / (dict[wordList[i]] + len(dict)))
                else:
                    logProba = math.log2((0 + 1) / (dict[wordList[i]] + len(dict)))
                sum += logProba

    if (undef):
        perplexity[type] = "undefined"
    else:
        perplexity[type] = 2 ** (- (1 / testTokens) * sum)

    File.close()

# log probability
def logProb(str, dict, trainDict, type, perplexity):
    line = ""
    if type == "unigram":
        for word in str.split():
            if word not in dict:
                dict[word] = 1
            else:
                dict[word] += 1

            if word not in trainDict:
                line += "<unk> "
                if "<unk>" not in dict:
                    dict["<unk>"] = 1
                else:
                    dict["<unk>"] += 1
            else:
                line += word + " "
        line = "<s> " + line + "</s>"
        dict["<s>"] = 1
        dict["</s>"] = 1
    else:
        for word in str.split():
            if word not in dict:
                line += "<unk> "
            else:
                line += word + " "
        line = "<s> " + line + "</s>"

    sum = 0
    undef = False
    if type == "unigram":
        for word in line.split():
            logProba = math.log2(trainDict[word] / wordTokens(trainDict))
            print("Log probability of", word + ":", logProba)
            sum += logProba
    elif type == "bigram":
        wordList = line.split()
        for i in range(len(wordList) - 1):
            biWords = (wordList[i], wordList[i + 1])
            logProba = 0
            if biWords in trainDict:
                logProba = math.log2(trainDict[biWords] / dict[wordList[i]])
                print("Log probability of", biWords.__str__() + ":", logProba)
            else:
                print("Log probability of", biWords.__str__() + ":", "undefined")
                undef = True
            sum += logProba
    else:
        wordList = line.split()
        for i in range(len(wordList) - 1):
            biWords = (wordList[i], wordList[i + 1])
            if biWords in trainDict:
                logProba = math.log2((trainDict[biWords] + 1) / (dict[wordList[i]] + len(dict)))
            else:
                logProba = math.log2((0 + 1) / (dict[wordList[i]] + len(dict)))
            print("Log probability of", biWords.__str__() + ":", logProba)
            sum += logProba

    if (undef):
        print("Summation of the inputs log probability: undefined")
        perplexity[type] = "undefined"
    else:
        print("Summation of the inputs log probability:", sum)
        perplexity[type] = 2 ** (- (1 / len(line.split())) * sum)
    print()

# counts words in txt file for unigram
def uniCountWords(txtFile, wordDict, wordDictWithoutUnk):
    File = open(txtFile, "r+")
    for line in File:
        for word in line.split():
            if word.lower() not in wordDict:
                wordDict[word.lower()] = 1
                wordDictWithoutUnk[word.lower()] = 1
            else:
                wordDict[word.lower()] += 1
                wordDictWithoutUnk[word.lower()] += 1

    File.close()

# counts words in processedTrain.txt for bigram
def biCountWords(txtFile, biDict):
    File = open(txtFile, "r+")
    for line in File:
        wordList = line.lower().split()
        for i in range(len(wordList) - 1):
            biWords = (wordList[i], wordList[i+1])
            if biWords not in biDict:
                biDict[biWords] = 1
            else:
                biDict[biWords] += 1

    File.close()

# adds <s>, </s> and <unk> tag
def addStartEndUnkTag(testF, trainF, trainingWords, trainingWordsWithoutUnk, testWords, testWordsWithoutUnk):
    trainingWords["<s>"] = 0
    trainingWords["</s>"] = 0
    trainingWords["<unk>"] = 0
    trainingWordsWithoutUnk["<s>"] = 0
    trainingWordsWithoutUnk["</s>"] = 0
    trainFile = open(trainF, "r+")
    processedTrainFile = open("processedTrain.txt", "w")
    for line in trainFile:
        newLine = ""
        for word in line.split():
            if trainingWords[word.lower()] == 1:
                newLine += " <unk>"
                trainingWords["<unk>"] += 1
                del trainingWords[word.lower()]
            else:
                newLine += " " + word

        processedTrainFile.writelines("<s>" + newLine.lower() + " </s>\n")
        trainingWords["<s>"] += 1
        trainingWords["</s>"] += 1
        trainingWordsWithoutUnk["<s>"] += 1
        trainingWordsWithoutUnk["</s>"] += 1

    trainFile.close()
    processedTrainFile.close()

    testWords["<s>"] = 0
    testWords["</s>"] = 0
    testWords["<unk>"] = 0
    testWordsWithoutUnk["<s>"] = 0
    testWordsWithoutUnk["</s>"] = 0
    testFile = open(testF, "r+")
    processedTestFile = open("processedTest.txt", "w")
    processedTestFile_without_unk = open("no_unk_processedTest.txt", "w")
    for line in testFile:
        newLine = ""
        for word in line.split():
            if word.lower() not in trainingWords:
                newLine += " <unk>"
                testWords["<unk>"] += 1
                if word.lower() in testWords:
                    del testWords[word.lower()]
            else:
                newLine += " " + word
        processedTestFile.writelines("<s>" + newLine.lower() + " </s>\n")
        processedTestFile_without_unk.writelines("<s> " + line.strip().lower() + " </s>\n")
        testWords["<s>"] += 1
        testWords["</s>"] += 1
        testWordsWithoutUnk["<s>"] += 1
        testWordsWithoutUnk["</s>"] += 1

    testFile.close()
    processedTestFile.close()
    processedTestFile_without_unk.close()

# word tokens in training corpus
def trainingWordTokens(trainingWords):
    count = 0
    for word in trainingWords:
        count += trainingWords[word]

    return count

# word tokens in test corpus
def wordTokens(wordDict):
    count = 0
    for word in wordDict:
        count += wordDict[word]

    return count

# returns % of word tokens in test not in train
def testTokensNotInTrain_Percent(testWords, trainingWords, type, testLineCount):
    if type == "unigram":
        return (testTokensNotInTrain(testWords, trainingWords, type) / (wordTokens(testWords) - testWords["<s>"])) * 100 # without <unk>, testWords not used here
    else:
        return (testTokensNotInTrain(testWords, trainingWords, type) / (wordTokens(testWords) - testLineCount)) * 100 # with <unk>, testLineCount for number of <s>

# returns % of word types in test not in train
def testTypesNotInTrain_Percent(testWords, trainingWords, type):
    count = 0
    testStartCount = 0
    for word in testWords:
        if word.__contains__("<s>"):
            testStartCount += 1
        if word not in trainingWords:
            count += 1

    if type == "unigram":
        return (count / (len(testWords) - 1)) * 100
    else:
        return (count / (len(testWords) - testStartCount)) * 100

def testTokensNotInTrain(testWords, trainingWords, type):
    count = 0

    if type == "unigram":
        no_unk_testFile = open("no_unk_processedTest.txt", "r+")
        for line in no_unk_testFile:
            line = line.lower()
            for word in line.split():
                if word not in trainingWords:
                    count += 1
        no_unk_testFile.close()
    else:
        for biWord in testWords:
            if biWord not in trainingWords:
                count += testWords[biWord]

    return count

main()