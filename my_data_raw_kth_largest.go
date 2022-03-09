package main

import (
	"container/heap"
	"sort"
)

type KthLargest struct {
	sort.IntSlice
	k int
}

//func Constructor(k int, nums []int) KthLargest {
//	kl := KthLargest{k: k}
//	for _, val := range nums {
//		kl.Add(val)
//	}
//	return kl
//}

func (kl *KthLargest) Push(v interface{}) {
	kl.IntSlice = append(kl.IntSlice, v.(int))
}

func (kl *KthLargest) Pop() interface{} {
	a := kl.IntSlice
	v := a[len(a)-1]
	kl.IntSlice = a[:len(a)-1]
	return v
}

func (kl *KthLargest) Add(val int) int {
	// pop 和 push必须要是大写的公有方法才能调用
	heap.Push(kl, val)
	if kl.Len() > kl.k {
		heap.Pop(kl)
	}
	// 堆？ 最大为root
	return kl.IntSlice[0]
}
