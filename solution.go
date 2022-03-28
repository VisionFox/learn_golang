package main

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
	"strconv"
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

func hammingWeight(num uint32) (ones int) {
	for i := 0; i < 32; i++ {
		if (1 << i & num) > 0 {
			ones++
		}
	}
	return
}

func kthSmallest(root *TreeNode, k int) int {
	stack := make([]*TreeNode, 0)
	for {
		for root != nil {
			stack = append(stack, root)
			root = root.Left
		}

		stack, root = stack[:len(stack)-1], stack[len(stack)-1]
		k--
		if k == 0 {
			return root.Val
		}

		root = root.Right
	}
}

func findMin(nums []int) int {
	left, right := 0, len(nums)-1
	for left < right {
		mid := left + (right-left)/2
		if nums[mid] > nums[right] {
			left = mid + 1
		} else if nums[mid] < nums[right] {
			right = mid
		} else {
			right--
		}
	}
	return nums[left]
}

func findLength(A []int, B []int) int {
	aLen, bLen := len(A), len(B)
	ans := 0

	// B不动，A滚动
	// A从i开始，B从0开始，取两节公共部分的重合处做对比
	for i := 0; i < aLen; i++ {
		len := min(bLen, aLen-i)
		maxLen := maxLength(A, i, B, 0, len)
		ans = max(maxLen, ans)
	}

	// A不动，B滚动
	// B从i开始，A从0开始，取两节公共部分的重合处做对比
	for i := 0; i < bLen; i++ {
		len := min(aLen, bLen-i)
		maxLen := maxLength(A, 0, B, i, len)
		ans = max(ans, maxLen)
	}

	return ans
}

// 公共部分的重复子串比较
func maxLength(A []int, aIdx int, B []int, bIdx, lenCommon int) int {
	ans := 0
	cnt := 0
	for i := 0; i < lenCommon; i++ {
		if A[aIdx+i] == B[bIdx+i] {
			cnt++
		} else {
			cnt = 0
		}
		ans = max(ans, cnt)
	}
	return ans
}

func getIntersectionNode(headA, headB *ListNode) *ListNode {
	tmpA, tmpB := headA, headB

	for tmpA != tmpB {
		if tmpA == nil {
			tmpA = headB
		} else {
			tmpA = tmpA.Next
		}

		if tmpB == nil {
			tmpB = headA
		} else {
			tmpB = tmpB.Next
		}

		//// 错误：转换也算走了一步！！！！
		//if tmpA == nil {
		//	tmpA = headB
		//}
		//if tmpB == nil {
		//	tmpB = headA
		//}
		//
		//tmpA = tmpA.Next
		//tmpB = tmpB.Next
	}
	return tmpB
}

func mergeTwoLists(list1 *ListNode, list2 *ListNode) *ListNode {
	preNode := &ListNode{Val: -1}
	tmp := preNode

	for list1 != nil && list2 != nil {
		if list1.Val < list2.Val {
			tmp.Next = list1
			list1 = list1.Next
		} else {
			tmp.Next = list2
			list2 = list2.Next
		}
		tmp = tmp.Next
	}

	if list1 != nil {
		tmp.Next = list1
	}

	if list2 != nil {
		tmp.Next = list2
	}

	return preNode.Next
}

func search(nums []int, target int) int {
	left, right := 0, len(nums)-1
	for left <= right {
		mid := left + (right-left)/2
		if nums[mid] > target {
			right = mid - 1
		} else if nums[mid] < target {
			left = mid + 1
		} else {
			return mid
		}
	}
	return -1
}

func twoSumV2(nums []int, target int) []int {
	m := make(map[int]int)
	for idx, n := range nums {
		tIdx, ok := m[target-n]
		if ok {
			return []int{tIdx, idx}
		} else {
			m[n] = idx
		}
	}
	return []int{-1, -1}
}

func sortArray(nums []int) []int {
	randomizedQuicksort(nums, 0, len(nums)-1)
	return nums
}

func randomizedQuicksort(nums []int, left, right int) {
	if left < right {
		pos := randomizedPartition(nums, left, right)
		randomizedQuicksort(nums, left, pos-1)
		randomizedQuicksort(nums, pos+1, right)
	}
}

func randomizedPartition(nums []int, left, right int) int {
	r := rand.Intn(right-left+1) + left
	swap(nums, r, right)
	return partition(nums, left, right)
}

func partition(nums []int, left, right int) int {
	pivot := nums[right]
	swapIdx := left - 1
	for j := left; j <= right-1; j++ {
		if nums[j] <= pivot {
			swapIdx++
			swap(nums, swapIdx, j)
		}
	}
	swap(nums, swapIdx+1, right)
	return swapIdx + 1
}

func maxSubArray(nums []int) int {
	dp := make([]int, len(nums))
	dp[0] = nums[0]
	ans := nums[0]
	for i := 1; i < len(nums); i++ {
		dp[i] = max(dp[i-1]+nums[i], nums[i])
		ans = max(dp[i], ans)
	}
	return ans
}

func reverseList(head *ListNode) *ListNode {
	if head == nil || head.Next == nil {
		return head
	}

	var pre *ListNode
	cur := head
	for cur != nil {
		next := cur.Next
		cur.Next = pre
		pre = cur
		cur = next
	}
	return pre
}

func reverseListV2(head *ListNode) *ListNode {
	if head == nil || head.Next == nil {
		return head
	}
	newHead := reverseList(head.Next)
	head.Next.Next = head
	head.Next = nil
	return newHead
}

func getKthFromEnd(head *ListNode, k int) *ListNode {
	fast, slow := head, head
	for fast != nil && k > 0 {
		fast = fast.Next
		k--
	}
	for fast != nil {
		fast = fast.Next
		slow = slow.Next
	}
	return slow
}

func largestNumber(nums []int) string {
	sort.Slice(nums, func(i, j int) bool {
		str1 := fmt.Sprintf("%d%d", nums[i], nums[j])
		str2 := fmt.Sprintf("%d%d", nums[j], nums[i])

		if str1 > str2 {
			return true
		} else {
			return false
		}
	})

	if nums[0] == 0 {
		return "0"
	}

	ans := ""
	for _, num := range nums {
		ans += strconv.Itoa(num)
	}
	return ans
}

func maxSlidingWindow(nums []int, k int) []int {
	if len(nums) < 2 {
		return nums
	}

	queue := make([]int, 0)
	ans := make([]int, len(nums)-k+1)
	for i := 0; i < len(nums); i++ {
		for len(queue) > 0 && nums[queue[len(queue)-1]] <= nums[i] {
			queue = queue[0 : len(queue)-1]
		}

		queue = append(queue, i)

		if queue[0] <= i-k {
			queue = queue[1:]
		}

		if i+1 >= k {
			ans[i+1-k] = nums[queue[0]]
		}
	}
	return ans
}

func preorderTraversal(root *TreeNode) []int {
	ans := make([]int, 0)

	var preorder func(node *TreeNode)

	preorder = func(node *TreeNode) {
		if node == nil {
			return
		}
		ans = append(ans, node.Val)
		preorder(node.Left)
		preorder(node.Right)
	}

	preorder(root)

	return ans
}

func preorderTraversalV2(root *TreeNode) []int {
	// 存个根
	stack := make([]*TreeNode, 0)
	ans := make([]int, 0)
	node := root

	for node != nil || len(stack) > 0 {
		for node != nil {
			ans = append(ans, node.Val)
			stack = append(stack, node)
			node = node.Left
		}

		node = stack[len(stack)-1].Right
		stack = stack[:len(stack)-1]
	}
	return ans
}

func deleteDuplicates(head *ListNode) *ListNode {
	if head == nil {
		return head
	}

	dummy := &ListNode{Val: 0, Next: head}

	cur := dummy
	for cur.Next != nil && cur.Next.Next != nil {
		if cur.Next.Val != cur.Next.Next.Val {
			cur = cur.Next
		} else {
			tmp := cur.Next.Val
			for cur.Next != nil && tmp == cur.Next.Val {
				cur.Next = cur.Next.Next
			}
		}
	}
	return dummy.Next
}

func addTwoNumbers(l1 *ListNode, l2 *ListNode) *ListNode {
	stack1 := make([]*ListNode, 0)
	stack2 := make([]*ListNode, 0)

	for l1 != nil {
		stack1 = append(stack1, l1)
		l1 = l1.Next
	}

	for l2 != nil {
		stack2 = append(stack2, l2)
		l2 = l2.Next
	}

	var ans *ListNode

	carry := 0
	for len(stack1) > 0 || len(stack2) > 0 {
		A := 0
		if len(stack1) > 0 {
			A = stack1[len(stack1)-1].Val
			stack1 = stack1[:len(stack1)-1]
		}

		B := 0
		if len(stack2) > 0 {
			B = stack2[len(stack2)-1].Val
			stack2 = stack2[:len(stack2)-1]
		}

		curNum := (A + B + carry) % 10
		carry = (A + B + carry) / 10

		ans = &ListNode{
			Val:  curNum,
			Next: ans,
		}
	}

	if carry > 0 {
		ans = &ListNode{
			Val:  carry,
			Next: ans,
		}
	}

	return ans
}

func reverseBetween(head *ListNode, left int, right int) *ListNode {
	dummy := &ListNode{Val: -1, Next: head}
	g, p := dummy, dummy.Next

	for i := 0; i < left-1; i++ {
		// g 走到left-1的位置
		g = g.Next
		// p 走到要逆转的位置
		p = p.Next
	}

	for i := 0; i < right-left; i++ {
		remove := p.Next

		p.Next = p.Next.Next
		remove.Next = g.Next
		g.Next = remove
	}

	return dummy.Next
}

func spiralOrder(matrix [][]int) []int {
	if len(matrix) == 0 {
		return make([]int, 0)
	}

	ans := make([]int, 0)
	up, down, left, right := 0, len(matrix)-1, 0, len(matrix[0])-1
	for true {
		for i := left; i <= right; i++ {
			ans = append(ans, matrix[up][i])
		}
		up++
		if up > down {
			break
		}

		for i := up; i <= down; i++ {
			ans = append(ans, matrix[i][right])
		}
		right--
		if right < left {
			break
		}

		for i := right; i >= left; i-- {
			ans = append(ans, matrix[down][i])
		}
		down--
		if down < up {
			break
		}

		for i := down; i >= up; i-- {
			ans = append(ans, matrix[i][left])
		}
		left++
		if left > right {
			break
		}
	}
	return ans
}

func nextPermutation(nums []int) {
	if len(nums) <= 1 {
		return
	}

	i, j, k := len(nums)-2, len(nums)-1, len(nums)-1

	for i >= 0 && nums[i] >= nums[j] {
		i--
		j--
	}

	if i >= 0 {
		for nums[i] >= nums[k] {
			k--
		}
		nums[i], nums[k] = nums[k], nums[i]
	}

	for i, j := j, len(nums)-1; i < j; i, j = i+1, j-1 {
		nums[i], nums[j] = nums[j], nums[i]
	}
}

func mySqrt(x int) int {
	if x == 0 || x == 1 {
		return x
	}

	left, right := 1, x/2
	// 在区间 [left..right] 查找目标元素
	for left < right {
		mid := left + (right-left+1)/2
		if mid*mid > x {
			// 下一轮搜索区间是 [left..mid - 1]
			right = mid - 1
		} else {
			// 下一轮搜索区间是 [mid..right]
			left = mid
		}
	}
	return left
}

func bstFromPreorder(preorder []int) *TreeNode {
	if len(preorder) == 0 {
		return nil
	}

	root := &TreeNode{Val: preorder[0]}
	preorder = preorder[1:]

	splitIdx := 0
	for i := 0; i < len(preorder); i++ {
		if preorder[i] > root.Val {
			break
		}
		splitIdx++
	}

	root.Left = bstFromPreorder(preorder[:splitIdx])
	root.Right = bstFromPreorder(preorder[splitIdx:])
	return root
}

func jump(nums []int) int {
	// 贪心：每次跳，都要选择这段区间内，在下一次能跳最远的地方
	maxPos, lastEnd, step := 0, 0, 0
	// 最后一个数不用管
	for i := 0; i < len(nums)-1; i++ {
		maxPos = max(maxPos, i+nums[i])
		if i == lastEnd {
			lastEnd = maxPos
			step++
		}
	}
	return step
}

func rotate(matrix [][]int) {
	size := len(matrix)

	for row := 0; row < size; row++ {
		for col := 0; col < row; col++ {
			temp := matrix[row][col]
			matrix[row][col] = matrix[col][row]
			matrix[col][row] = temp
		}
	}

	for row := 0; row < size; row++ {
		for left, right := 0, size-1; left < right; {
			temp := matrix[row][left]
			matrix[row][left] = matrix[row][right]
			matrix[row][right] = temp
			left++
			right--
		}
	}
}

func find132pattern(nums []int) bool {
	// 1:minIdx 3:maxIdx 2:midIdx
	// 从后往前遍历的单调递减栈：存储的是：2：midIdx:maxValue
	// 而弹出的是3:maxIdx:midValue
	// 遍历时，当有3:midValue > 1:minValue时则满足题意
	stack := make([]int, 0)
	midValue := math.MinInt

	for i := len(nums) - 1; i >= 0; i-- {
		if midValue > nums[i] {
			return true
		}

		for len(stack) > 0 && stack[len(stack)-1] < nums[i] {
			midValue = max(midValue, stack[len(stack)-1])
			stack = stack[:len(stack)-1]
		}
		stack = append(stack, nums[i])
	}

	return false
}

func canCompleteCircuit(gas []int, cost []int) int {
	for idxStart := 0; idxStart < len(gas); idxStart++ {
		idxEnd, remain := idxStart, gas[idxStart]
		for remain-cost[idxEnd] >= 0 {
			idxNext := (idxEnd + 1) % len(gas)
			remain -= cost[idxEnd] + gas[idxNext]
			idxEnd = idxNext

			if idxNext == idxStart {
				return idxStart
			}
		}

		if idxEnd < idxStart {
			return -1
		}

		idxStart = idxEnd + 1
	}
	return -1
}

func permuteUnique(nums []int) [][]int {
	var res [][]int
	used := make([]bool, len(nums))
	sort.Ints(nums)
	helper([]int{}, nums, used, &res)
	return res
}

func helper(path, nums []int, used []bool, res *[][]int) {
	if len(path) == len(nums) {
		temp := make([]int, len(nums))
		copy(temp, path)
		*res = append(*res, temp)
		return
	}

	for i := 0; i < len(nums); i++ {
		if i-1 >= 0 && nums[i-1] == nums[i] && !used[i-1] {
			continue
		}
		if used[i] {
			continue
		}

		path = append(path, nums[i])
		used[i] = true

		helper(path, nums, used, res)
		path = path[0 : len(path)-1]

		used[i] = false
	}
}

func findBottomLeftValue(root *TreeNode) int {
	queue := make([]*TreeNode, 0)
	queue = append(queue, root)

	var leftest *TreeNode
	for len(queue) > 0 {
		levelSize := len(queue)
		leftest = queue[0]

		for levelSize > 0 {
			// 取尾错误
			//node := queue[len(queue)-1]
			//queue = queue[:len(queue)-1]
			// 应该取头
			node := queue[0]
			queue = queue[1:]

			levelSize--

			if node.Left != nil {
				queue = append(queue, node.Left)
			}
			if node.Right != nil {
				queue = append(queue, node.Right)
			}
		}
	}
	return leftest.Val
}

func diffWaysToCompute(input string) []int {
	// 如果是数字，直接返回
	if isDigit(input) {
		tmp, _ := strconv.Atoi(input)
		return []int{tmp}
	}

	// 空切片
	var ans []int
	// 遍历字符串
	for index, c := range input {
		tmpC := string(c)
		if tmpC == "+" || tmpC == "-" || tmpC == "*" {
			// 如果是运算符，则计算左右两边的算式
			left := diffWaysToCompute(input[:index])
			right := diffWaysToCompute(input[index+1:])

			for _, leftNum := range left {
				for _, rightNum := range right {
					var addNum int
					if tmpC == "+" {
						addNum = leftNum + rightNum
					} else if tmpC == "-" {
						addNum = leftNum - rightNum
					} else {
						addNum = leftNum * rightNum
					}
					ans = append(ans, addNum)
				}
			}
		}
	}

	return ans
}

func findMedianSortedArrays(nums1 []int, nums2 []int) float64 {
	left := (len(nums1) + len(nums2) + 1) / 2
	right := (len(nums1) + len(nums2) + 2) / 2
	return (findKth(nums1, 0, nums2, 0, left) + findKth(nums1, 0, nums2, 0, right)) / 2.0
}

func findKth(nums1 []int, idx1 int, nums2 []int, idx2 int, k int) float64 {
	if idx1 >= len(nums1) {
		return float64(nums2[idx2+k-1])
	}
	if idx2 >= len(nums2) {
		return float64(nums1[idx1+k-1])
	}

	if k == 1 {
		return float64(min(nums1[idx1], nums2[idx2]))
	}

	mid1Idx := idx1 + k/2 - 1
	mid1Value := math.MaxInt

	mid2Idx := idx2 + k/2 - 1
	mid2Value := math.MaxInt

	if mid1Idx < len(nums1) {
		mid1Value = nums1[mid1Idx]
	}

	if mid2Idx < len(nums2) {
		mid2Value = nums2[mid2Idx]
	}

	if mid1Value < mid2Value {
		return findKth(nums1, mid1Idx+1, nums2, idx2, k-k/2)
	} else {
		return findKth(nums1, idx1, nums2, mid2Idx+1, k-k/2)
	}
}

func maxProfit(k int, prices []int) int {
	if 2*k > len(prices) {
		return maxProfitGreedy(prices)
	}

	return maxProfitDp(k, prices)
}

func maxProfitDp(k int, prices []int) int {
	buys := make([]int, k)
	sells := make([]int, k)
	for i := 0; i < k; i++ {
		buys[i] = -prices[0]
	}

	ans := 0

	for i := 1; i < len(prices); i++ {
		price := prices[i]
		for j := 0; j < k; j++ {
			if j == 0 {
				buys[j] = max(buys[j], 0-price)
			} else {
				buys[j] = max(buys[j], sells[j-1]-price)
			}
			sells[j] = max(sells[j], buys[j]+price)
			ans = max(sells[j], ans)
		}
	}

	return ans
}

func maxProfitGreedy(prices []int) int {
	ans := 0
	for i := 1; i < len(prices); i++ {
		if prices[i] > prices[i-1] {
			ans += prices[i] - prices[i-1]
		}
	}
	return ans
}
