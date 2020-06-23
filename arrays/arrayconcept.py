n = input("Enter number of integers you want:")
a = list()
print("Enter integers of the array you wish to create:")
for i in range(int(n)):
    print("Number ",i+1,":",end="")
    a.append(input())
print("Integer array:",a)
n = len(a)
x = input("Enter element to be inserted:")
pos = input("Enter 1 for inserting at end, 2 for some other position:")
print(pos)
if int(pos) == 1:
    a.append(x)
    print("Array after insertion:",a)
else:
    pos1 = input("Enter position at which you want to insert element:")
    n = len(a)    
    a.insert(int(pos1)-1,x)
            
    print("Array after insertion:",a)
delete = input("Enter element you want to delete:")
a.remove(delete)
print("Array after deletion:",a)
