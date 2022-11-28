package handler

import (
	"math/rand"
	"sort"
)

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
