import random
import math
import matplotlib.pyplot as plt
personNum=1000 #种群数量
mutationProbability=0.1 #变异概率
iteration=500 #假设迭代50次即终止
length=30

def getAbsList(theList):
    for i in range(len(theList)):
        if theList[i]<0:
            theList[i]=theList[i]*(-1)
    return theList

# 功能：生成初始化种群
# 参数：personNum为种群数量，length为种群每个个体编码的位数
def initialPopulation(personNum=50,length=30):
    totalPopulation=[]
    while len(totalPopulation)!=personNum:
        person=[]
        for i in range(length):
            temp = random.uniform(-1, 1)  # 生成-1<=X<=1的数字
            if temp<0:
                person.append(0)
            else:
                person.append(1)
        theStr = ''
        for item in person:
            theStr += str(item)
        #print(theStr)
        if theStr not in totalPopulation:
            if evaluate(theStr,length)>0:
                totalPopulation.append(theStr)
        #print(len(totalPopulation))
    return totalPopulation


# 函数功能：将编码转换为x,y的十进制解
def decode(onePerson,length=30):
    index=0
    data=[]
    while index<length:
        #print(onePerson[index:index+15])
        data.append(int(onePerson[index:index+15],2)*math.pi/32678)
        index+=15
    #print("没报错")
    return data


# 功能：计算x,y对应的函数值
# 参数：一个个体的编码
def evaluate(onePerson,length=30):
    data=decode(onePerson,length)
    result=0
    for i in range(int(length/15)):
        x=data[i]
        result+=math.sin(x) * math.pow(math.sin((i+1)*x * x / math.pi), 20)
    #print(result)
    return result

# 功能：获取一个父母进行交叉
# 输出：返回的是一个双亲在population的index
def getParents(evalList):
    temp = random.uniform(0, 1)
    #print(temp)
    portionList=[];theSum=0
    totalEval=sum(evalList)
    #print(totalEval)
    for eval in evalList:
        theSum+=eval/totalEval
        portionList.append(theSum)
    location=0
    while(temp>portionList[location]):
        location+=1
    #print('location=',location)
    return location


# 输入：两个person
# 输出：生成的子代person编码
def getCross(father,mother,length=30):
    theVisit=[]
    crossLocation=random.randint(0,length-1)
    theVisit.append(crossLocation)
    #print(crossLocation)
    child=''
    child += father[0:crossLocation]
    child += mother[crossLocation:length]
    while evaluate(child,length)<0:
        #print("重新交叉")
        while crossLocation in theVisit and len(theVisit)<length:
            crossLocation = random.randint(0, length-1)
            #print(crossLocation)
            child += father[0:crossLocation]
            child += mother[crossLocation:]
        theVisit.append(crossLocation)
        if len(theVisit)>=length:
            child=father
        #print(len(child))
    return child


# 功能：进行变异
def getVari(person,length=30):
    #print(person)
    temp = random.uniform(0, 1)
    if temp<mutationProbability:
        #print('变异')
        location=random.randint(0,length-1)
        #print(location)
        tempStr=person[0:location]
        tempStr+=str(1-int(person[location]))
        tempStr+=person[location+1:]
        if evaluate(tempStr)>evaluate(person):
            return tempStr
    return person


if __name__=='__main__':
    theScore=[]
    bestPerson=[]
    theBestEval=0
    print("输入d的大小：")
    d=int(input())
    length=d*15
    population = initialPopulation(personNum,length)
    flag = 0
    bestRecord=[]
    while flag!=iteration:
        print("第",flag+1,"代")
        evalList=[]
        tempPopulation=[]
        for person in population:
            evalList.append(evaluate(person,length))
        maxEval=max(evalList)
        print('maxEval=',maxEval)
        theIndex=evalList.index(maxEval)
        tempPopulation.append(population[theIndex]) #每次迭代时先将上一代最大的个体放到下一代种群中
       # print("开始交叉")
        for i in range(personNum):
            #获得用于交叉的父母
            parentsFaIndex=getParents(evalList)
            parentsFa=population[parentsFaIndex]
            parentsMaIndex=getParents(evalList)
            parentsMa=population[parentsMaIndex]
            child=getCross(parentsFa,parentsMa,length)

            child=getVari(child,length)
            tempPopulation.append(child)
        population=tempPopulation
        flag+=1

        evalList = []
        for person in population:
            #print(person)
            evalList.append(evaluate(person,length))
        maxEval=max(evalList)

        if theBestEval<maxEval:
            theBestEval=maxEval
        theIndex = evalList.index(maxEval)
        person = population[theIndex]
        if person not in bestPerson:
            bestPerson.append(person)
            theScore.append(1)
        else:
            theScore[bestPerson.index(person)] += 1
        bestRecord.append(-theBestEval)
    # print('duration=',time.time()-timeStart)

    print(theScore)
    print(bestPerson)
    theBestEvalList=[]
    for item in bestPerson:
        theBestEvalList.append(evaluate(item,length))
    print(theBestEvalList)
    print(theBestEval)
    print(max(theScore))
    print(bestRecord)
    iterations=[i for i in range(iteration)]
    plt.plot(iterations, bestRecord, 'bo-')
    plt.title("Convergence curve")
    plt.show()