"""
Code Contributed by --
Anubhav Natani
"""
import re
import pandas as pd
# script generates a new data file that contains all the data cleaned
datas = pd.read_csv("data.csv")
data = []


def featuresFind(s, i):
    result = re.findall('(\\n==(.*?)==\\n)', s)
    a = []
    for i in result:
        if(i[1][0] != "="):
            a.append(i[1])
    data.append(a)


for i in range(0, 293):
    s = datas.iloc[i, 2]
    featuresFind(s, i)
a = {}
for i in data:
    for j in range(0, len(i)):
        try:
            a[i[j]] = a[i[j]]+1
        except:
            a[i[j]] = 1
finalFeatures = []
for key, value in a.items():
    if value > 100:
        finalFeatures.append(key)
newFinalFeatures = [' Signs and symptoms ', ' Causes ', ' Pathophysiology ',
                    ' Diagnosis ', ' Prognosis ', ' Treatment ', ' Prevention ']


def subtopicFinder(s, subtopics, topic):
    result = re.findall('(===(.*?)===)', s)
    a = []
    for i in result:
        a.append(i[1])
    subtopics[topic] = a


newData = []


def codeSegerator(i):
    newArr = {}
    b = datas.iloc[i, 2]
    b = b.split("\n\n\n== ")
    for i in range(0, len(b)):
        b[i] = b[i].split(" ==\n")
    newArr['About'] = b[0][0]
    subtopics = {}
    for i in b:
        if(len(i) == 2):
            if(" "+i[0]+" " in newFinalFeatures):
                a = i[1].replace("\n", " ")
                subtopicFinder(a, subtopics, i[0])
                bn = re.sub('(===(.*?)===)', " ", a)
                newArr[i[0]] = bn

    newArr['Subtopic'] = subtopics
    newData.append(newArr)


for i in range(0, 293):
    codeSegerator(i)

for i in range(0, 293):
    newData[i]["DiseaseName"] = datas.iloc[i, 1]

df = pd.DataFrame(newData)

df.to_csv("dataSep.csv", index=False)
