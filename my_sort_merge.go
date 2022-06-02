package main

func mergeSort(nums []int) []int {
	length := len(nums)
	if length < 2 {
		return nums
	}
	middle := length / 2
	leftNums := nums[:middle]
	rightNums := nums[middle:]

	//mergeSort(leftNums)
	//mergeSort(rightNums)

	return mergeTemp(leftNums, rightNums)
}

func mergeTemp(leftNums, rightNums []int) []int {
	ans := make([]int, 0)

	if len(leftNums) > 0 && len(rightNums) > 0 {
		if leftNums[0] <= rightNums[0] {
			ans = append(ans, leftNums[0])
			leftNums = leftNums[1:]
		} else {
			ans = append(ans, rightNums[0])
			rightNums = rightNums[1:]
		}
	}

	for len(leftNums) > 0 {
		ans = append(ans, leftNums[0])
		leftNums = leftNums[1:]
	}

	for len(rightNums) > 0 {
		ans = append(ans, rightNums[0])
		rightNums = rightNums[1:]
	}

	return ans
}
