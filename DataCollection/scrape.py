"""
Code Contributed by --
Anubhav Natani
"""
import os
import wikipedia
import pandas as pd
diseaseList = []
filePath = os.path.join(os.getcwd(), "common_disease.txt")
with open(filePath, "r") as fp:
    diseaseList = fp.readlines()
diseaseList = [i[:-1] for i in diseaseList]
diseaseList[-1] = diseaseList[-1]+"r"

diseaseName = []
diseaseNameWiki = []
dataExtract = []
for i in range(0, len(diseaseList)):
    query = diseaseList[i]
    # wikipedia api
    try:
        final_query = wikipedia.search(query, results=1)
        final_query = final_query[0]
        resp = wikipedia.page(final_query)
        diseaseName.append(query)
        diseaseNameWiki.append(final_query)
        dataExtract.append(resp.content)
    except:
        print(query)

data = pd.DataFrame(data={"diseaseName": diseaseName,
                          "diseaseWikiName": diseaseNameWiki, "FullInfo": dataExtract})
data.to_csv(index=False)
