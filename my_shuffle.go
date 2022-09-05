package main

import "math/rand"

type MyShuffle struct {
	original []int
	nums     []int
}

func NewMyShuffle(nums []int) MyShuffle {
	return MyShuffle{
		original: nums,
		nums:     append([]int(nil), nums...),
	}
}

func (ms *MyShuffle) Reset() []int {
	copy(ms.nums, ms.original)
	return ms.nums
}

func (ms *MyShuffle) Shuffle() []int {
	n := len(ms.original)
	for idx1, _ := range ms.nums {
		idx2 := idx1 + rand.Intn(n-idx1)
		ms.nums[idx1], ms.nums[idx2] = ms.nums[idx2], ms.nums[idx1]
	}
	return ms.nums
}
