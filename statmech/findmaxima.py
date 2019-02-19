f=open("/Users/syedather/Desktop/highdata.txt", "r")
a=[]
def FindMaxima(numbers):
    maxima = []
    length = len(numbers)
    if length >= 2:
        if numbers[0] > numbers[1]:
            maxima.append(numbers[0])

        if length > 3:
            for i in range(1, length-1):
                if numbers[i] > numbers[i-1] and numbers[i] > numbers[i+1]:
                    maxima.append(numbers[i])

        if numbers[length-1] > numbers[length-2]:
            maxima.append(numbers[length-1])
    return maxima
for i in f.readlines():
    j=float(i.replace("\n", ""))
    a.append(j)
#print a
b=FindMaxima(a)
c=[]
for i in a:
    if i in b:
        c.append(i)
    else:
        c.append(float(0))
print c