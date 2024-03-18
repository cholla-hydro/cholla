with open('cluster_list_MW_low.txt','r') as ofile:
    data = ofile.read()

data = data.replace('\t',',')
data = data.replace('\n',',\n')
data = data.replace('#','//')
#print(data)

with open('cluster_list_MW.data','w') as ofile:
    ofile.write('Real cluster_data[] = {\n')
    ofile.write(data)
    ofile.write('};\n')
