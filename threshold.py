
List = open("/home/salih/Desktop/test/output_of_val2017.txt").readlines()

for line in List:
    lime = line[17:20]
    
    
    lks = float(lime)
    #print(lks)
    
    if(lks > 0.6):
        print(line)
        #line = line[:25] + "0 " + line[26:]
        with open('/home/salih/Desktop/test/output_of_val2017thre06.txt','a') as f:
            f.write(line)
