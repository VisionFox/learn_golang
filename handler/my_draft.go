package handler

import (
	"errors"
	"sort"
)

var OPERATION1FAILED error = errors.New("1")
var OPERATION2FAILED error = errors.New("2")
var OPERATION3FAILED error = errors.New("3")
var OPERATION4FAILED error = errors.New("4")

func Handle() error {
	var err error
	if Operation1() {
		if Operation2() {
			if Operation3() {
				if Operation4() {
					// do
				} else {
					err = OPERATION4FAILED
				}
			} else {
				err = OPERATION3FAILED
			}
		} else {
			err = OPERATION2FAILED
		}
	} else {
		err = OPERATION1FAILED
	}
	return err
}

func HandlePlus() error {
	var err error

	for true {
		if !Operation1() {
			err = OPERATION1FAILED
			break
		}

		if !Operation2() {
			err = OPERATION1FAILED
			break
		}

		if !Operation3() {
			err = OPERATION1FAILED
			break
		}

		if !Operation4() {
			err = OPERATION1FAILED
			break
		}

		// do

		break
	}

	return err
}

func Operation1() bool {
	return false
}

func Operation2() bool {
	return false
}

func Operation3() bool {
	return false
}

func Operation4() bool {
	return false
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
