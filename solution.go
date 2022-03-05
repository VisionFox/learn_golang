package main

import (
	"fmt"
	"sort"
)

func findContentChildren(g []int, s []int) int {
	sort.Ints(g)
	sort.Ints(s)
	indexG, indexS := 0, 0

	for indexG < len(g) && indexS < len(s) {
		if g[indexG] <= s[indexS] {
			indexG++
		}
		indexS++
	}
	return indexG
}

func candy(ratings []int) int {
	res := make([]int, len(ratings))

	for idx := 0; idx < len(ratings); idx++ {
		if idx > 0 && ratings[idx] > ratings[idx-1] {
			res[idx] = res[idx-1] + 1
		} else {
			res[idx] = 1
		}
	}

	for idx := len(ratings) - 1; idx > 0; idx-- {
		if ratings[idx-1] > ratings[idx] {
			res[idx-1] = maxInt(res[idx-1], res[idx]+1)
		}
	}

	cnt := 0
	for _, i := range res {
		cnt += i
	}
	return cnt
}

func maxInt(i, j int) int {
	if i > j {
		return i
	}
	return j
}

func eraseOverlapIntervals(intervals [][]int) int {
	if len(intervals) == 0 {
		return 0
	}

	sort.Slice(intervals, func(i, j int) bool {
		return intervals[i][1] < intervals[j][1]
	})

	removeCnt := 0
	tailFlag := intervals[0][1]
	for i := 1; i < len(intervals); i++ {
		if intervals[i][0] < tailFlag {
			removeCnt++
		} else {
			tailFlag = intervals[i][1]
		}
	}

	return removeCnt
}

func findMinArrowShots(points [][]int) int {
	if len(points) == 0 {
		return 0
	}

	fmt.Printf("%v\n", points)

	sort.Slice(points, func(i, j int) bool {
		return points[i][1] < points[j][1]
	})

	fmt.Printf("%v\n", points)

	cnt := 1
	tailFlag := points[0][1]

	for i := 1; i < len(points); i++ {
		if points[i][0] > tailFlag {
			cnt++
			tailFlag = points[i][1]
		}
	}

	return cnt
}

func partitionLabels(s string) []int {
	ans := make([]int, 0)
	var marks [26]int
	size, left, right := len(s), 0, 0
	for i := 0; i < size; i++ {
		marks[s[i]-'a'] = i
	}

	for i := 0; i < size; i++ {
		right = maxInt(right, marks[s[i]-'a'])
		if i == right {
			ans = append(ans, right-left+1)
			left = right + 1
		}
	}

	return ans
}

func reconstructQueue(people [][]int) [][]int {
	ans := make([][]int, 0)
	fmt.Printf("%v\n", people)
	sort.Slice(people, func(i, j int) bool {
		if people[i][0] != people[j][0] {
			// 下降
			return people[i][0] > people[j][0]
		}
		// 上升
		return people[i][1] < people[j][1]
	})

	fmt.Printf("%v\n", people)

	for _, p := range people {
		insertIdx := p[1]
		ans = append(ans[:insertIdx], append([][]int{p}, ans[insertIdx:]...)...)
	}
	fmt.Printf("%v\n", ans)

	return ans
}

func twoSum(numbers []int, target int) []int {
	left, right := 0, len(numbers)-1

	for left < right {
		sum := numbers[left] + numbers[right]
		if sum == target {
			return []int{left + 1, right + 1}
		} else if sum > target {
			right--
		} else {
			left++
		}
	}

	return []int{-1, -1}
}

func merge(nums1 []int, m int, nums2 []int, n int) {
	for m > 0 && n > 0 {
		if nums2[n-1] >= nums1[m-1] {
			nums1[m+n-1] = nums2[n-1]
			n--
		} else {
			nums1[m+n-1] = nums1[m-1]
			m--
		}
	}

	for n > 0 {
		nums1[n-1] = nums2[n-1]
		n--
	}
}

func detectCycleHash(head *ListNode) *ListNode {
	nodePtrSet := make(map[*ListNode]struct{})
	for head != nil {
		if _, exist := nodePtrSet[head]; exist {
			return head
		}
		nodePtrSet[head] = struct{}{}
		head = head.Next
	}
	return nil
}

func detectCycle(head *ListNode) *ListNode {
	fast, slow := head, head
	for fast != nil {
		slow = slow.Next
		if fast.Next == nil {
			return nil
		}
		fast = fast.Next.Next

		if fast == slow {
			tmp := head
			for tmp != slow {
				slow = slow.Next
				tmp = tmp.Next
			}
			return tmp
		}
	}
	return nil
}
