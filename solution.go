package main

import (
	"fmt"
	"math"
	"math/rand"
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
			res[idx-1] = max(res[idx-1], res[idx]+1)
		}
	}

	cnt := 0
	for _, i := range res {
		cnt += i
	}
	return cnt
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
		right = max(right, marks[s[i]-'a'])
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

func minWindow(s string, t string) string {
	// 统计t字符串内的情况
	targetExist := make(map[byte]bool)
	targetCnt := make(map[byte]int)
	for i := 0; i < len(t); i++ {
		targetExist[t[i]] = true
		targetCnt[t[i]]++
	}

	// 滑动窗口
	windowsContainerCnt := 0
	minLeft, minSize := 0, len(s)+1
	for left, right := 0, 0; right < len(s); right++ {
		if targetExist[s[right]] {
			targetCnt[s[right]]--
			if targetCnt[s[right]] >= 0 {
				windowsContainerCnt++
			}

			// 若目前滑动窗口已包含T中全部字符，
			// 则尝试将l右移， 在不影响结果的情况下获得最短子字符串
			for windowsContainerCnt == len(t) {
				newSize := right - left + 1
				if newSize < minSize {
					minLeft = left
					minSize = newSize
				}

				if targetExist[s[left]] {
					targetCnt[s[left]]++
					if targetCnt[s[left]] > 0 {
						windowsContainerCnt--
					}
				}

				left++
			}
		}
	}

	if minSize > len(s) {
		return ""
	} else {
		return s[minLeft : minLeft+minSize]
	}
}

func judgeSquareSum(c int) bool {
	left, right := 0, int(math.Sqrt(float64(c)))

	for left <= right {
		sum := left*left + right*right
		if sum == c {
			return true
		} else if sum > c {
			right--
		} else {
			left++
		}
	}
	return false
}

func validPalindrome(s string) bool {
	left, right := 0, len(s)-1
	for left <= right {
		if s[left] == s[right] {
			left++
			right--
		} else {
			return validPalindromeCheck(s, left+1, right) || validPalindromeCheck(s, left, right-1)
		}
	}
	return true
}

func validPalindromeCheck(s string, left, right int) bool {
	for left <= right {
		if s[left] != s[right] {
			return false
		} else {
			left++
			right--
		}
	}
	return true
}

func uniquePaths(m int, n int) int {
	dp := make([][]int, m)
	for i := 0; i < m; i++ {
		dp[i] = make([]int, n)
		dp[i][0] = 1
	}

	for i := 0; i < n; i++ {
		dp[0][i] = 1
	}

	for i := 1; i < m; i++ {
		for j := 1; j < n; j++ {
			dp[i][j] = dp[i-1][j] + dp[i][j-1]
		}
	}

	return dp[m-1][n-1]
}

func findRepeatNumber(nums []int) int {
	set := make(map[int]struct{})
	for i := 0; i < len(nums); i++ {
		if _, exist := set[nums[i]]; exist {
			return nums[i]
		} else {
			set[nums[i]] = struct{}{}
		}
	}
	return -1
}

func plusOne(head *ListNode) *ListNode {
	var cur, first_9 *ListNode = head, nil

	for cur != nil {
		if cur.Val != 9 {
			first_9 = cur
		}
		cur = cur.Next
	}

	if first_9 != nil {
		first_9.Val++
		first_9 = first_9.Next
		for first_9 != nil {
			first_9.Val = 0
		}
		return head
	}

	newHead := &ListNode{
		Val:  1,
		Next: head,
	}

	for head != nil {
		head.Val = 0
		head = head.Next
	}
	return newHead
}

func partitionDisjoint(nums []int) int {
	leftMax := make([]int, len(nums))
	rightMin := make([]int, len(nums))

	m := math.MinInt
	for i := 0; i < len(nums); i++ {
		m = max(m, nums[i])
		leftMax[i] = m
	}

	m = math.MaxInt
	for i := len(nums) - 1; i >= 0; i-- {
		m = min(m, nums[i])
		rightMin[i] = m
	}

	for i := 1; i < len(nums); i++ {
		if leftMax[i-1] <= rightMin[i] {
			return i
		}
	}
	return -1
}

func partitionDisjointV2(nums []int) int {
	leftMax, maxs := nums[0], nums[0]
	leftPos := 0
	for i := 1; i < len(nums); i++ {
		maxs = max(maxs, nums[i])
		if nums[i] >= leftMax {
			continue
		}
		leftMax = max(leftMax, maxs)
		leftPos = i
	}
	return leftPos + 1
}

type Solution struct {
	pre []int
}

//func Constructor(w []int) Solution {
//	// 计算前缀和
//	pre := make([]int, len(w))
//	pre[0] = w[0]
//	for i := 1; i < len(w); i++ {
//		pre[i] = pre[i-1] + w[i]
//	}
//
//	return Solution{
//		pre: pre,
//	}
//}

func (s *Solution) PickIndex() int {
	// 注意，生成的随机数不能包含0，否则部分用例过不了
	r := rand.Intn(s.pre[len(s.pre)-1]) + 1
	// 判断rand的范围落在哪个区间
	// Search(len(a), func(i int) bool { return a[i] >= x })
	// return i
	return sort.SearchInts(s.pre, r)
}

type Trie struct {
	children [26]*Trie
	isEnd    bool
}

func Constructor() Trie {
	return Trie{}
}

func (t *Trie) Insert(word string) {
	node := t
	for _, ch := range word {
		ch -= 'a'
		if node.children[ch] == nil {
			node.children[ch] = &Trie{}
		}
		node = node.children[ch]
	}
	node.isEnd = true
}

func (t *Trie) Search(word string) bool {
	node := t.searchPrefix(word)
	return node != nil && node.isEnd == true
}

func (t *Trie) StartsWith(prefix string) bool {
	return t.searchPrefix(prefix) != nil
}

func (t *Trie) searchPrefix(prefix string) *Trie {
	node := t
	for _, ch := range prefix {
		ch -= 'a'
		if node.children[ch] == nil {
			return nil
		}
		node = node.children[ch]
	}
	return node
}
