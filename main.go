package main

import "learn_golang/handler"

func main() {
	handler.MyPrintCatDogFish()
}

//func tQSort(arr []int, left, right int) []int {
//	if left >= right {
//		return arr
//	}
//
//	p := tpartition(arr, left, right)
//	tQSort(arr, left, p-1)
//	tQSort(arr, p+1, right)
//	return arr
//}

//func tpartition(arr []int, left, right int) int {
//	basePivot := left
//	scanIdx := basePivot + 1
//	for i := scanIdx; i <= right; i++ {
//		if arr[i] < arr[basePivot] {
//			handler.Swap(arr, scanIdx, i)
//			scanIdx++
//		}
//	}
//
//	handler.Swap(arr, scanIdx-1, basePivot)
//	return scanIdx - 1
//}

//func tMSort(arr []int, left, right int) []int {
//	if left == right {
//		return arr
//	}
//	mid := left + (right-left)/2
//	tMSort(arr, left, mid)
//	tMSort(arr, mid+1, right)
//	tmerge(arr, left, mid, right)
//	return arr
//}
//
//func tmerge(arr []int, left, mid, right int) []int {
//	p1, p2 := left, mid+1
//	tmpArr := make([]int, 0)
//	for p1 <= mid && p2 <= right {
//		if arr[p1] > arr[p2] {
//			tmpArr = append(tmpArr, arr[p2])
//			p2++
//		} else {
//			tmpArr = append(tmpArr, arr[p1])
//			p1++
//		}
//	}
//
//	for p1 <= mid {
//		tmpArr = append(tmpArr, arr[p1])
//		p1++
//	}
//
//	for p2 <= right {
//		tmpArr = append(tmpArr, arr[p2])
//		p2++
//	}
//
//	for i := 0; i < len(tmpArr); i++ {
//		arr[left] = tmpArr[i]
//		left++
//	}
//
//	return arr
//}
