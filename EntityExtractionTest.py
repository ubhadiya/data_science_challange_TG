import re
import pandas as pd
import numpy as np


df = pd.read_csv("/Users/kamlesh/Documents/Technical Data/Python_example/PythonFirst/TechGig/test.csv")
fwrite=open("/Users/kamlesh/Documents/Technical Data/Python_example/PythonFirst/TechGig/output.csv","a")
df=df.replace(r'\n',"")

df = df [['id','description']]
heading=''.join(['id',',','StringToExtract'])
fwrite.write(heading)
fwrite.write("\n")

num_lines = len(df)

def writeToFile(id,stringToExtract):
    filewrite=''.join([str(id),',',str(stringToExtract)])
    fwrite.write(filewrite)
    fwrite.write("\n")

ciscoIpAddress=("10.200.3.37","10.200.3.165","cisco ip phone")
withoutIpNode=("asia.","controlnet.","nyc-core.","uk.ecnahcdroffilc","eu.","-dsw-","-dsw1","-asw-")


for i in range(num_lines):
    df.iloc[i,1]=re.sub(R'\s\r\n',"\s",str(df.iloc[i,1]))
    output=re.search(r'cisco ipt alert.*item:([\s]*[\w-]+)', df.iloc[i,1])
    if not output:
        if "criticalservicedown" in df.iloc[i,1] and "client service" in df.iloc[i,1] and not(any(ip in df.iloc[i,1] for ip in ciscoIpAddress)):
            output = re.search(r'(service)', df.iloc[i,1])
        else:
            output=re.search(r'(?:cisco ipt alert|(?:internal.ifrdoflc)).*?(\d{1,3}[.]\d{1,3}[.]\d{1,3}[.]\d{1,3})', df.iloc[i,1])
    if not output:
        output=re.search(r'([\w-]+) on (?:vnx7600|vnx5600|vnx5400)', df.iloc[i,1])
    if not output:
        output=re.search(r'vnx system [\']?([\w-]+)', df.iloc[i,1])
    if not output:
        output=re.search(r'storage system ([\w-]+) of type vnx', df.iloc[i,1])
    else:
        output=output.group(1).strip()
        output=re.sub(r'[:\';//]',"",output)
        if not output:
            writeToFile(df.iloc[i,0],output)
        else:
            writeToFile(df.iloc[i,0],output)
        continue
    if not output and "system will automatically clean" in df.iloc[i,1]:
        output=re.search(r'(will)', df.iloc[i,1])
    if not output:
        output=re.search(r'library.zone:(\s?[\w-]+)', df.iloc[i,1])
    if not output:
        output=re.search(r'against the computer(\s?[\w-]+)[.]', df.iloc[i,1])
    if not output:
        output=re.search(r'failed.*?client:(\s?[\w-]+)\s', df.iloc[i,1])
    if not output:
        output=re.search(r'commvault alert.*?(?:library|mediaagent) name:(\s?[\w-]+)', df.iloc[i,1])
    if not output and "gpmszfd1" not in df.iloc[i,1]:
         output=re.search(r'target.name:(\s?[\w-]+)', df.iloc[i,1])
    if not output:
        output=re.search(r'target.host.*?:(\s?[\w-]+)', df.iloc[i,1])
    if not output and not(any(region in df.iloc[i,1] for region in withoutIpNode)):
        output=re.search(r'(?:node|interface goes down|(?:advanced alert)).*?(\d{1,3}[.]\d{1,3}[.]\d{1,3}[.]\d{1,3}).*?roalswinds.oionr', df.iloc[i,1])
    if not output and "hkg-sim" not in df.iloc[i,1]:
         output=re.search(r'(?:sim critical|minor|major alarm).*?(\d{1,3}[.]\d{1,3}[.]\d{1,3}[.]\d{1,3})', df.iloc[i,1])
    if not output:
         output=re.search(r'(?:operations nageram failed to start).*?(\d{1,3}[.]\d{1,3}[.]\d{1,4}[.]\d{1,3})', df.iloc[i,1])
    if not output:
        output=re.search(r'(?<!h server )[:\'//,\s"]([\w=-]+[.](internal|intranet|ecnahcdroffilc|inf|ld|asia|fra|hkg|pm|bei|per|buc|tok|controlnet|bru|americas|muc|germany|leaorc).*(net|com)[;\s\':/])', df.iloc[i,1])
    else:
        output=output.group(1).strip()
        output=re.sub(r'[:\';//]',"",output)
        if not output:
            writeToFile(df.iloc[i,0],output)
        else:
            writeToFile(df.iloc[i,0],output)
        continue
    if not output:
        output=re.search(r'[\s]+([\w-]+)[\s]+(failed|was not)', df.iloc[i,1])
    if not output:
        output=re.search(r'(will)', df.iloc[i,1])
    if not output:
        writeToFile(df.iloc[i,0],output)
    else:
        output=output.group(1).strip()
        output=re.sub(r'[:\';//]',"",output)
        output=re.search(r'^([\w=-]+)[.]?', output)
        if not output:
            writeToFile(df.iloc[i,0],output)
        else:
            output=output.group(1).strip()
            if output == "device":
                output="error" 
            writeToFile(df.iloc[i,0],output)
    i=i+1
fwrite.close()
