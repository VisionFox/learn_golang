package main

import (
	"fmt"
	"learn_golang/handler"
)

func main() {
	handler.SayHello()

	a := []int{1, 2, 4}
	println(candy(a))
	b := [][]int{{7, 0}, {4, 4}, {7, 1}, {5, 0}, {6, 1}, {5, 2}}
	println(reconstructQueue(b))
	strA := "ADOBECODEBANC"
	strB := "ABC"
	println(minWindow(strA, strB))
	handler.SayHello()

	fmt.Printf("%v\n", restoreIpAddresses("25525511135"))
}
