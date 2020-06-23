
def optimal_pagerep(s,capacity):
    occurance = [None for i in range(capacity)]
    f = []
    fault = 0
    for i in range(len(s)):
        if s[i] not in f:
            if len(f)<capacity:
                f.append(s[i])
            else:
                for x in range(len(f)):
                    if f[x] not in s[i+1:]:
                        f[x] = s[i]
                        break
                    else:
                        occurance[x] = s[i+1:].index(f[x])
                else:
                    f[occurance.index(max(occurance))] = s[i]
            fault += 1
            pf = 'PF'
        else:
            pf = 'Hit'
        print(s[i],":",end='')
        for x in f:
            print(x,end=' ')
        
        print(pf)
    print("Total Page Faults:",fault)

capacity = 4
s = ['7', '0', '1', '2', '0', '3', '0', '16', '2', '3']
optimal_pagerep(s,capacity)
