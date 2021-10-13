with open('cluster_list.txt','r') as ofile:
    data = ofile.read()

data = data.replace('\t',',')
data = data.replace('\n',',\n')
data = data.replace('#','//')
#print(data)

with open('cluster_list.data','w') as ofile:
    ofile.write('Real cluster_data[] = {\n')
    ofile.write(data)
    ofile.write('};\n')
