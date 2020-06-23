
def partition(arr,low,high): 
    i = ( low-1 )         
    pivot = arr[high]     
    print("Pivot:",pivot)
    for j in range(low , high): 
   
        if arr[j] <= pivot: 
            print(arr)
            i = i+1 
            print("Swapping",arr[i],"and",arr[j])
            arr[i],arr[j] = arr[j],arr[i] 
    print("Swapping",arr[i+1],"and",arr[high])   
    arr[i+1],arr[high] = arr[high],arr[i+1] 
    print(arr)
    return ( i+1 ) 
  
def quickSort(arr,low,high): 
    if low < high:  
        pi = partition(arr,low,high) 
        
        quickSort(arr, low, pi-1) 
        quickSort(arr, pi+1, high) 
  
# Driver code to test above 
arr = ['8', '9', '14', '7']
n = len(arr) 
quickSort(arr,0,n-1) 
print ("Sorted array is:") 
print(arr)