package main

import "math/rand"

func randomizedQuicksort(nums []int, left, right int) {
	if left < right {
		splitPos := randomizedPartition(nums, left, right)
		randomizedQuicksort(nums, left, splitPos-1)
		randomizedQuicksort(nums, splitPos+1, right)
	}
}

func randomizedPartition(nums []int, left, right int) int {
	r := rand.Intn(right-left+1) + left
	swap(nums, r, left)
	return partition(nums, left, right)
}

func partition(nums []int, left, right int) int {
	pivot := nums[left]
	leftMark := left
	rightMark := right

	for leftMark != rightMark {
		// 一直找到右边小于或者等于主元 不符合分配的位置
		for leftMark < rightMark && nums[rightMark] > pivot {
			rightMark--
		}
		// 一直找到右边大于主元 不符合分配的位置
		for leftMark < rightMark && nums[leftMark] <= pivot {
			leftMark++
		}
		// 交换
		if leftMark < rightMark {
			swap(nums, leftMark, rightMark)
		}
	}
	// 交换主元
	swap(nums, leftMark, left)

	return leftMark
}
