package main

import "sort"

func main() {
	test([][]int{{2, 4}, {3, 5}, {5, 7}})
}

func test(events [][]int) {
	// 1,start up ,
	// 2 ,start == start, end up
	sort.Slice(events, func(i, j int) bool {
		if events[i][0] != events[j][0] {
			return events[i][0] < events[j][0]
		}

		if events[i][0] == events[j][0] {
			return events[i][1] < events[i][1]
		}

		return false
	})

	flagStart := events[0][0]
	flagEnd := events[0][1]

	println("%v---%v", flagStart, flagEnd)

	for i := 1; i < len(events); i++ {
		start, end := events[i][0], events[i][1]

		// 冲突
		if start < flagEnd {
			continue
		}

		// 选排序的第一个
		if flagStart == start {
			continue
		}

		// 不冲突
		println("%v---%v", start, end)

		flagStart = start
		flagEnd = end
	}
}
