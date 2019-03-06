# -*-coding:utf-8-*-


# 记录快排程序
def QuickSort(arr, firstIndex, lastIndex):
    if firstIndex < lastIndex:  # 终止条件
        divIndex = Partition(arr, firstIndex, lastIndex)

        QuickSort(arr, firstIndex, divIndex)  # 递归调用前半部分
        QuickSort(arr, divIndex+1, lastIndex)  # 递归调用后半部分
    else:
        return


def Partition(arr, firstIndex, lastIndex):
    i = firstIndex - 1
    for j in range(firstIndex, lastIndex):
        if arr[j] <= arr[lastIndex]:
            i = i+1  # i记录前面有几个数比lastIndex的小，第一次执行Partition函数可让以arr[lastIndex]为中间值来分成两部分
            arr[i], arr[j] = arr[j], arr[i]
    arr[i+1], arr[lastIndex] = arr[lastIndex], arr[i+1]  # 将arr[lastIndex]的数据放在该放的地方，此时可保证前后两部分对arr[i+1]来说相对有序
    return i


if __name__ == '__main__':
    arr = [1, 4, 7, 1, 5, 5, 3, 85, 34, 75, 23, 75, 2, 0]
    print("initial array: \n", arr)
    QuickSort(arr, 0, len(arr)-1)
    print("result array: \n", arr)
